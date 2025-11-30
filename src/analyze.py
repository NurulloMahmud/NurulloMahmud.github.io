"""
Analyze module - Business analysis functions

This module answers the key business questions:
1. What are the top 5 days by revenue?
2. How many unique users are there (after deduplication)?
3. How many unique author sets exist?
4. Who is the most popular author?
5. Who is the top customer?
6. What is the daily revenue trend?
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Set


# SECTION 1: Daily Revenue Functions

def calculate_daily_revenue(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily revenue from orders.
    
    Args:
        orders_df: Transformed orders DataFrame (must have 'date' and 'paid_price' columns)
    
    Returns:
        DataFrame with columns: date, revenue (sorted by date)
    """
    # group by date and sum the paid_price
    daily = orders_df.groupby('date')['paid_price'].sum().reset_index()
    
    # rename columns for clarity
    daily.columns = ['date', 'revenue']
    
    # sort by date (asc order)
    daily = daily.sort_values('date')
    
    # round revenue to 2 decimal places
    daily['revenue'] = daily['revenue'].round(2)
    
    return daily


def get_top_revenue_days(orders_df: pd.DataFrame, n: int = 5) -> List[Dict]:
    """
    Get top N days by revenue.
    
    Args:
        orders_df: Transformed orders DataFrame
        n: Number of top days to return (default 5)
    
    Returns:
        List of dicts: [{'date': 'YYYY-MM-DD', 'revenue': float}, ...]
        Sorted by revenue descending.
    """
    # calculate daily revenue
    daily = calculate_daily_revenue(orders_df)
    
    # get top N days (largest revenue values)
    top_days = daily.nlargest(n, 'revenue')
    
    # convert to list of dictionaries for JSON serialization
    return [
        {'date': row['date'], 'revenue': round(row['revenue'], 2)}
        for _, row in top_days.iterrows()
    ]


# SECTION 2: Union-Find Data Structure (for User Deduplication)

class UnionFind:
    """
    Example usage:
        uf = UnionFind()
        uf.union(101, 102) -> User 101 and 102 are same person
        uf.union(102, 103) -> User 102 and 103 are same person
        • Now 101, 102, 103 are all in the same group
        
        uf.find(103) -> Returns 101 (the group representative)
        uf.get_groups() -> Returns {101: {101, 102, 103}}
    """
    
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def find(self, x):
        """
        Find the root (representative) of the group containing x.
        Args:
            x: The element to find the root for
        Returns:
            The root element of the group containing x
        """
        # If x is not yet in our structure, add it as its own group
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        
        # path compression: if x's parent is not the root, recursively find root, and make x point directly to root
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
    
    def union(self, x, y):
        """
        Merge the groups containing x and y.
        """
        # find the roots of both elements
        root_x = self.find(x)
        root_y = self.find(y)
        
        # if they're already in the same group, nothing to do
        if root_x == root_y:
            return
        
        # union by rank: attach smaller tree under larger tree
        if self.rank[root_x] < self.rank[root_y]:
            # root_y's tree is taller, so make it the parent
            root_x, root_y = root_y, root_x
        
        # make root_x the parent of root_y
        self.parent[root_y] = root_x
        
        # if both trees had same rank, the combined tree is one level taller
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
    
    def get_groups(self) -> Dict[int, Set[int]]:
        """
        Get all groups as a dictionary.
        
        Returns:
            Dict mapping root_id -> set of all member ids in that group
        """
        groups = defaultdict(set)
        
        # for each element we've seen, find its root and add to that group
        for x in self.parent:
            root = self.find(x)
            groups[root].add(x)
        
        return dict(groups)


# SECTION 3: User Deduplication

def deduplicate_users(users_df: pd.DataFrame) -> Tuple[int, Dict[int, List[int]]]:
    """
    Deduplicate users based on matching 3+ out of 4 fields.
    
    According to requirements: "user can change address or change phone or 
    even provide alias instead of a real name... only one field is changed."
    
    This means if only 1 field changed, then 3 fields must be the same.
    We only count matches for fields where BOTH users have non-null values.
    
    Returns:
        Tuple: (unique_user_count, {canonical_id: [all_user_ids]})
        
    Example:
        If users 101, 102, 103 are the same person:
        Returns: (1, {101: [101, 102, 103]})
    """
    uf = UnionFind()
    
    # step 1: add all user IDs to Union-Find
    # this initializes each user as their own group
    for uid in users_df['id']:
        uf.find(uid)
    
    # step 2: Build indexes for faster lookup
    # instead of comparing every user with every other user 
    # we only compare users that share at least one field (potential matches)
    by_email = defaultdict(list)
    by_phone = defaultdict(list)
    by_name = defaultdict(list)
    by_address = defaultdict(list)
    
    for _, row in users_df.iterrows():
        uid = row['id']
        
        # Only index non-null, non-empty values
        if pd.notna(row.get('email_normalized')) and row['email_normalized']:
            by_email[row['email_normalized']].append(uid)
        if pd.notna(row.get('phone_normalized')) and row['phone_normalized']:
            by_phone[row['phone_normalized']].append(uid)
        if pd.notna(row.get('name_normalized')) and row['name_normalized']:
            by_name[row['name_normalized']].append(uid)
        if pd.notna(row.get('address_normalized')) and row['address_normalized']:
            by_address[row['address_normalized']].append(uid)
    
    # step 3: create a lookup for user data (for comparing fields later)
    user_data = {}
    for _, row in users_df.iterrows():
        user_data[row['id']] = {
            'email': row.get('email_normalized'),
            'phone': row.get('phone_normalized'),
            'name': row.get('name_normalized'),
            'address': row.get('address_normalized')
        }
    
    # step 4: find potential matches (users that share at least one field)
    # we use a set to avoid checking the same pair twice
    potential_pairs = set()
    
    # users with same email might be same person
    for users_list in by_email.values():
        if len(users_list) > 1:
            for i, u1 in enumerate(users_list):
                for u2 in users_list[i+1:]:
                    # store as sorted tuple to avoid duplicates like (101,102) and (102,101)
                    potential_pairs.add((min(u1, u2), max(u1, u2)))
    
    # users with same phone might be same person
    for users_list in by_phone.values():
        if len(users_list) > 1:
            for i, u1 in enumerate(users_list):
                for u2 in users_list[i+1:]:
                    potential_pairs.add((min(u1, u2), max(u1, u2)))
    
    # users with same name might be same person
    for users_list in by_name.values():
        if len(users_list) > 1:
            for i, u1 in enumerate(users_list):
                for u2 in users_list[i+1:]:
                    potential_pairs.add((min(u1, u2), max(u1, u2)))
    
    # users with same address might be same person
    for users_list in by_address.values():
        if len(users_list) > 1:
            for i, u1 in enumerate(users_list):
                for u2 in users_list[i+1:]:
                    potential_pairs.add((min(u1, u2), max(u1, u2)))
    
    # step 5: check each potential pair
    # if they match on 3+ fields (where both have values), union them
    for u1, u2 in potential_pairs:
        d1 = user_data[u1]
        d2 = user_data[u2]
        
        matches = 0
        
        # count matching fields, only count matches for fields where both users have non-null values
        for field in ['email', 'phone', 'name', 'address']:
            v1 = d1.get(field)
            v2 = d2.get(field)
            
            if v1 and v2:
                if v1 == v2:
                    matches += 1
        
        # if they match on 3+ fields, they're the same person
        if matches >= 3:
            uf.union(u1, u2)
    
    # step 6: get final groups
    groups = uf.get_groups()
    
    # convert to expected format: {canonical_id: [sorted list of member ids]}
    # we use the smallest id as the canonical id
    user_groups = {
        min(members): sorted(list(members)) 
        for members in groups.values()
    }
    
    unique_count = len(groups)
    
    return unique_count, user_groups


# SECTION 4: Author Analysis

def count_unique_author_sets(books_df: pd.DataFrame) -> Tuple[int, List[str]]:
    """
    Count unique author sets.
    
    Each distinct combination of authors counts as one set.
    Example: 'John' alone, 'John, Paul' together, and 'Paul' alone = 3 sets.
    
    Args:
        books_df: Transformed books DataFrame with 'author_set' column
                  (contains frozensets of author names)
    
    Returns:
        Tuple: (count, list of author set strings for display)
    """
    # collect unique author sets, frozensets can be added to a set (they're hashable)
    unique_sets = set()
    
    for author_set in books_df['author_set']:
        if author_set is not None:
            unique_sets.add(author_set)
    
    # Convert frozensets to readable strings for display | frozenset({'john', 'jane'}) -> "jane, john" (sorted alphabetically)
    author_set_strings = [
        ', '.join(sorted(list(aset))) 
        for aset in unique_sets
    ]
    
    return len(unique_sets), sorted(author_set_strings)


def find_most_popular_author(orders_df: pd.DataFrame, books_df: pd.DataFrame) -> Tuple[str, int]:
    """
    Find the most popular author (or author set) by total books sold.
    
    "Most popular" = highest sum of quantity across all orders.
    
    Args:
        orders_df: Transformed orders DataFrame (with book_id, quantity)
        books_df: Transformed books DataFrame (with id, author, author_set)
    
    Returns:
        Tuple: (author_name_or_set_string, total_books_sold)
    """
    # step 1: join orders with books
    merged = orders_df.merge(
        books_df[['id', 'author', 'author_set']], 
        left_on='book_id', 
        right_on='id',
        how='left'
    )
    
    # step 2: convert frozenset to string for grouping
    merged['author_set_str'] = merged['author_set'].apply(
        lambda x: ', '.join(sorted(list(x))) if x else 'Unknown'
    )
    
    # step 3: group by author_set and sum quantities sold
    author_sales = merged.groupby('author_set_str')['quantity'].sum().reset_index()
    author_sales.columns = ['author', 'total_sold']
    
    # sstep 4: find the author(s) with most sales
    top = author_sales.nlargest(1, 'total_sold').iloc[0]
    
    # step 5: format author names nicely (Title Case)
    author_display = ', '.join([
        name.strip().title() 
        for name in top['author'].split(',')
    ])
    
    return author_display, int(top['total_sold'])


# SECTION 5: Top Customer Analysis

def find_top_customer(orders_df: pd.DataFrame, user_groups: Dict[int, List[int]]) -> Tuple[List[int], float]:
    """
    Find the top customer by total spending.
    
    If the customer has multiple user_ids (aliases), return all of them.
    
    Args:
        orders_df: Transformed orders DataFrame (with user_id, paid_price)
        user_groups: Dictionary from deduplicate_users {canonical_id: [all_ids]}
    
    Returns:
        Tuple: (list_of_all_user_ids, total_spending)
        
    Example:
        If user 101 and 102 are same person, and together they spent $800:
        Returns: ([101, 102], 800.00)
    """
    # step 1: calculate spending per user_id
    user_spending = orders_df.groupby('user_id')['paid_price'].sum().to_dict()
    
    # step 2: Create reverse mapping: user_id -> canonical_id, this tells us which "real person" each user_id belongs to
    user_to_canonical = {}
    for canonical, members in user_groups.items():
        for uid in members:
            user_to_canonical[uid] = canonical
    
    # step 3: aggregate spending by canonical user (the "real person")
    canonical_spending = defaultdict(float)
    for uid, spending in user_spending.items():
        # find which real person this user_id belongs to
        canonical = user_to_canonical.get(uid, uid)
        canonical_spending[canonical] += spending
    
    # step 4: find the top spender
    top_canonical = max(canonical_spending, key=canonical_spending.get)
    total_spent = canonical_spending[top_canonical]
    
    # step 5: get all user_ids for this person
    all_ids = user_groups.get(top_canonical, [top_canonical])
    
    return sorted(all_ids), round(total_spent, 2)


# SECTION 6: Main Orchestrator

def run_analysis(users_df: pd.DataFrame, orders_df: pd.DataFrame, books_df: pd.DataFrame) -> Dict:
    """
    Run all analysis and return results as a dictionary.
    
    This is the main entry point for analysis. It calls all other functions
    and returns a dictionary ready for JSON serialization.
    """
    daily_revenue = calculate_daily_revenue(orders_df)
    top_5_days = get_top_revenue_days(orders_df, n=5)
    
    unique_users_count, user_groups = deduplicate_users(users_df)
    
    unique_author_sets_count, author_sets_list = count_unique_author_sets(books_df)
    
    most_popular_author, books_sold = find_most_popular_author(orders_df, books_df)
    
    top_customer_ids, total_spent = find_top_customer(orders_df, user_groups)
    
    return {
        'top_5_revenue_days': top_5_days,
        'unique_users_count': unique_users_count,
        'unique_author_sets_count': unique_author_sets_count,
        'most_popular_author': most_popular_author,
        'most_popular_author_books_sold': books_sold,
        'top_customer_ids': top_customer_ids,
        'top_customer_total_spent': total_spent,
        'daily_revenue': daily_revenue.to_dict('records')
    }