# Bookstore ETL Pipeline

A complete ETL (Extract, Transform, Load) pipeline for analyzing bookstore sales data, with a professional BI-style dashboard.

##   Live Dashboard

**[View the Dashboard](https://NurulloMahmud.github.io/NurulloMahmud.github.io/dashboard/)**

---

##   Project Structure
```
bookstore-etl/
├── config/
│   └── settings.py          # Configuration settings
├── src/
│   ├── extract.py           # Load data from CSV, Parquet, YAML
│   ├── transform.py         # Clean and transform data
│   ├── analyze.py           # Business analysis functions
│   └── visualize.py         # Matplotlib chart generation
├── data/
│   ├── DATA1/               # Dataset 1
│   ├── DATA2/               # Dataset 2
│   └── DATA3/               # Dataset 3
├── output/
│   ├── charts/              # Generated PNG charts
│   └── results/             # JSON analysis results
├── dashboard/
│   ├── index.html           # Interactive dashboard
│   └── data/                # Dashboard data files
├── main.py                  # Pipeline orchestrator
├── requirements.txt         # Python dependencies
└── README.md
```

---

##   Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/bookstore-etl.git
cd bookstore-etl

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py
```

---

##   Features

### Data Processing
- **Multi-format support**: CSV, Parquet, YAML
- **Smart price parsing**: Handles `$27.00`, `€50¢50`, `USD 45.99`, `EUR26.99`
- **Flexible date parsing**: Handles 20+ timestamp formats
- **Currency conversion**: EUR → USD at 1.2 rate

### Analysis
1. **Daily Revenue**: Top 5 days by revenue
2. **User Deduplication**: Identifies same users across records (matching 3+ of 4 fields)
3. **Author Sets**: Counts unique author combinations
4. **Most Popular Author**: By total books sold
5. **Top Customer**: By total spending, with all associated user IDs

### Dashboard
- Interactive 3-tab interface (one per dataset)
- Professional BI-style design
- Daily revenue line charts
- Responsive Bootstrap layout

---

##   Output Format

### Top 5 Revenue Days
```json
[
  {"date": "2024-12-17", "revenue": 57011.46},
  {"date": "2024-11-03", "revenue": 46656.31}
]
```

### Best Buyer
```json
{
  "top_customer_ids": [45800],
  "top_customer_total_spent": 37609.70
}
```

---

##   Dependencies

- pandas
- pyarrow
- pyyaml
- python-dateutil
- matplotlib
- numpy

---
