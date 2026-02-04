# Elevating Cinnamon Hotels & Resorts in Sri Lanka: Using Guest Reviews and Arrival Data


## 📋 Project Overview

This project analyzes guest reviews from Booking.com and TripAdvisor, combined with arrival data from the Sri Lanka Tourism Development Authority (SLTDA), to provide actionable insights for improving service quality across **12 Cinnamon Hotels & Resorts properties in Sri Lanka**. Using **sentiment analysis (VADER & TextBlob)**, **topic modeling (LDA)**, and **predictive forecasting (Prophet)**, the project delivers a comprehensive yearly playbook and interactive dashboard for 2027.

**Student:** M.M. Senul Laksindu Semal (w2000706)  
**Supervisor:** Mrs. Sapna Dissanayake  
**Institution:** Informatics Institute of Technology / University of Westminster  
**Programme:** BSc (Hons) Data Science and Analytics  
**Academic Year:** 2025/2026  

---

## 🎯 Project Objectives

The primary aim is to analyze guest reviews and arrival data for each Cinnamon Hotels & Resorts property individually, predict trends and demands for 2027, and develop a comprehensive playbook containing tailored predictions and improvement recommendations for each hotel to achieve operational excellence.

**Specific objectives:**

1. Collecting and preprocessing review data on a per-hotel basis to ensure quality and consistency
2. Performing exploratory analysis to identify hotel-specific trends, complaints, and key satisfaction drivers
3. Applying sentiment analysis and topic modeling to generate targeted insights for each property
4. Developing playbooks with prioritized actions, predictive forecasts for 2027 (e.g., occupancy spikes and complaint patterns), and improvement strategies based on integrated data per hotel
5. Creating a visual dashboard to monitor key performance indicators (KPIs), track improvements, and visualize 2027 predictions over time

---

## 📊 Dataset Information

### Review Data
- **Booking.com:** 11,049+ reviews across 12 properties
- **TripAdvisor:** 23,934+ reviews across 12 properties
- **Total Reviews:** ~35,000

### Arrival Data
- **Source:** Sri Lanka Tourism Development Authority (SLTDA)
- **Type:** Monthly tourist arrival statistics merged with review data
- **Purpose:** Correlation analysis between occupancy levels and service quality perceptions

### Properties Covered (12 Cinnamon Hotels)
1. Cinnamon Grand Colombo
2. Cinnamon Lakeside Colombo
3. Cinnamon Red Colombo
4. Cinnamon Life at City of Dreams Colombo
5. Kandy Myst by Cinnamon
6. Cinnamon Bey Beruwala
7. Hikka Tranz by Cinnamon
8. Trinco Blu by Cinnamon
9. Cinnamon Lodge Habarana
10. Habarana Village by Cinnamon
11. Cinnamon Citadel Kandy
12. Cinnamon Wild Yala

---

## 📂 Repository Structure

```
cinnamon-hotels-analysis/
│
├── notebooks/
│   ├── 1_FYP-BookingDataCleaning_Preprocessing.ipynb
│   ├── 2_FYP-BookingDataMerging.ipynb
│   ├── 3_FYP-TripAdvisorDataCleaning_Preprocessing.ipynb
│   ├── 4_FYP-TripAdvisorDataMerging.ipynb
│   ├── 5_FYP-ArrivalDataMerging.ipynb
│   ├── 6_FYP-ReviewsMerging_DownloadingBadReviews.ipynb
│   ├── 7_FYP-BadReviewInspection.ipynb
│   ├── 8_FYP-BadReviewAnalysis.ipynb
│   └── 9_FYP-FurtherBadReviewAnalysiswithArrivalData.ipynb
│
├── visualizations/
│   ├── rating_distributions.png
│   ├── sentiment_trends.png
│   ├── correlation_heatmap.png
│   ├── wordcloud_overall.png
│   └── (other generated plots)
│
├── data/
│   ├── raw/ (Note: Raw data not included due to size/privacy)
│   └── processed/ (Cleaned datasets)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Technologies & Tools Used

### Programming Language
- **Python 3.8+**

### Data Processing & Analysis
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **openpyxl** - Excel file handling

### Visualization
- **matplotlib** - Static visualizations
- **seaborn** - Statistical data visualization
- **plotly** - Interactive plots
- **wordcloud** - Word cloud generation

### Natural Language Processing (NLP)
- **NLTK** - Natural Language Toolkit (tokenization, stopwords, lemmatization)
- **langdetect** - Language detection for filtering non-English reviews
- **vaderSentiment** - Sentiment intensity analysis
- **TextBlob** - Sentiment polarity and subjectivity

### Machine Learning
- **scikit-learn** - LDA topic modeling, TF-IDF vectorization
- **CountVectorizer, TfidfVectorizer** - Text feature extraction

### Time Series Forecasting
- **Prophet** - Facebook's forecasting tool for arrival and complaint predictions

### Web Scraping
- **Apify** - Web scraper API for Booking.com and TripAdvisor data extraction

### Dashboard (In Development)
- **Power BI** - Interactive dashboard for management (expected completion: March 2026)

### Development Environment
- **Google Colab** - Cloud-based Jupyter notebook environment
- **Jupyter Notebook** - Local development

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or Google Colab
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/cinnamon-hotels-analysis.git
cd cinnamon-hotels-analysis
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Required NLTK Data
Run this in Python:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
```

### Step 4: Run Notebooks Sequentially
Open Jupyter Notebook:
```bash
jupyter notebook
```

Execute notebooks in order (1-9) for complete analysis pipeline.

---

## 📓 Notebook Descriptions

| # | Notebook Name | Description |
|---|---------------|-------------|
| **1** | `FYP-BookingDataCleaning_Preprocessing.ipynb` | Cleans and preprocesses 11,000+ Booking.com reviews for all 12 properties. Handles missing values, standardizes columns, converts data types. |
| **2** | `FYP-BookingDataMerging.ipynb` | Merges individual hotel Booking.com datasets into a unified master file with quality checks. |
| **3** | `FYP-TripAdvisorDataCleaning_Preprocessing.ipynb` | Processes 24,000+ TripAdvisor reviews. Normalizes 5-point ratings to 10-point scale, removes HTML tags and encoding issues. |
| **4** | `FYP-TripAdvisorDataMerging.ipynb` | Consolidates TripAdvisor data across all properties with temporal verification. |
| **5** | `FYP-ArrivalDataMerging.ipynb` | Integrates SLTDA arrival statistics with review data for occupancy-based correlation analysis. |
| **6** | `FYP-ReviewsMerging_DownloadingBadReviews.ipynb` | Merges Booking.com and TripAdvisor data; filters negative reviews (rating ≤3.0) for focused analysis. |
| **7** | `FYP-BadReviewInspection.ipynb` | Comprehensive EDA on bad reviews including language detection, text preprocessing, temporal trends, rating distributions, traveler segmentation, and word frequency analysis. |
| **8** | `FYP-BadReviewAnalysis.ipynb` | Applies VADER sentiment analysis and aspect-based classification (staff, cleanliness, facilities, etc.). Implements LDA topic modeling. |
| **9** | `FYP-FurtherBadReviewAnalysiswithArrivalData.ipynb` | Advanced analysis combining sentiment trends with arrival patterns to identify occupancy-complaint correlations. |

---

## 📈 Key Findings (Preliminary)

### Most Common Complaint Categories
- **Staff behavior** - Rudeness, slow service, lack of professionalism
- **Cleanliness issues** - Room hygiene, bathroom conditions
- **WiFi connectivity** - Slow internet, frequent disconnections
- **Facilities** - Maintenance issues, outdated amenities
- **Value for money** - Price vs. service quality mismatch

### Insights by Analysis Type

**Temporal Trends:**
- Peak complaint periods align with high-occupancy months (December-March)
- Post-monsoon periods show increased facility-related complaints

**Property-Specific Patterns:**
- Urban hotels (Colombo properties) receive more WiFi and service speed complaints
- Resort properties receive more facility and activity-related complaints
- Beach resorts show seasonal maintenance complaint spikes

**Traveler Segmentation:**
- **Families:** More facility and cleanliness complaints
- **Business travelers:** Focus on WiFi, service efficiency, and value
- **Couples:** Emphasize ambiance and romantic experience quality

**Sentiment Analysis:**
- Negative sentiment scores correlate strongly with aspect-specific ratings
- VADER sentiment compound scores range from -0.8 (extremely negative) to -0.1 (mildly negative)

**Occupancy Correlation:**
- Higher occupancy periods (>80% capacity) show 25-30% increase in negative feedback
- Service quality degradation most visible in housekeeping and front desk aspects

---

## 🔧 Usage Guide

### Running Individual Notebooks

Each notebook is designed to be self-contained but should be executed sequentially for the complete analysis pipeline.

**Example - Running Sentiment Analysis (Notebook 8):**
```python
# Import VADER sentiment analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize analyzer
analyzer = SentimentIntensityAnalyzer()

# Apply sentiment analysis
df['sentiment_score'] = df['review_text'].apply(
    lambda x: analyzer.polarity_scores(x)['compound'] if x.strip() else 0
)

# View results
print(df[['review_text', 'sentiment_score']].head())
```

**Example - Topic Modeling with LDA:**
```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Create document-term matrix
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(df['clean_text'])

# Apply LDA
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_topics = lda_model.fit_transform(doc_term_matrix)

# Display top words per topic
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda_model.components_):
    top_words = [feature_names[i] for i in topic.argsort()[-10:]]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")
```

### Generating Visualizations

All visualizations are automatically saved in the `visualizations/` folder when executing notebooks. Example outputs include:
- Rating distribution histograms
- Sentiment trend line plots
- Correlation heatmaps
- Word clouds
- Hotel comparison box plots

---

## 📊 Dashboard (In Development)

An interactive **Power BI dashboard** is currently under development, featuring:

### Dashboard Features
- **Real-time KPI monitoring** - Overall ratings, sentiment scores, complaint volumes
- **Sentiment trend analysis** - By property, time period, traveler type
- **Topic distribution heatmaps** - Visual representation of complaint categories
- **Occupancy vs. complaint correlations** - Interactive scatter plots
- **Predictive forecasts for 2027** - Prophet-based predictions with confidence intervals
- **Property comparison tools** - Side-by-side performance metrics
- **Drill-down capabilities** - From portfolio level to individual hotel insights

**Expected Completion:** March 2026

---

## 🔄 Project Timeline

### Completed (October - December 2025)
- ✅ Data acquisition from Booking.com and TripAdvisor using Apify
- ✅ Data cleaning and preprocessing (Notebooks 1, 3)
- ✅ Data merging and integration (Notebooks 2, 4, 5, 6)
- ✅ Exploratory Data Analysis (Notebook 7)
- ✅ Sentiment analysis and topic modeling (Notebooks 8, 9)

### In Progress (January - March 2026)
- 🔄 Dashboard development in Power BI
- 🔄 Predictive modeling for 2027 forecasts
- 🔄 Yearly playbook creation with prioritized recommendations
- 🔄 Final report and presentation preparation

---

## 🚧 Challenges Encountered

### 1. Web Scraping Reliability
- **Issue:** Booking.com and TripAdvisor implement anti-bot measures (CAPTCHA, rate limiting, IP bans)
- **Solution:** Switched to Apify web scraper with proxy rotation, randomized delays, and anti-detection features

### 2. Platform Data Inconsistencies
- **Issue:** Different rating scales (Booking: 10-point, TripAdvisor: 5-point), inconsistent field names, missing reviewer location data
- **Solution:** Standardized scales (divided Booking ratings by 2), renamed columns uniformly, excluded Agoda due to missing location data

### 3. Multilingual Review Content
- **Issue:** ~10-15% of reviews in non-English languages
- **Solution:** Applied `langdetect` library to filter English-only reviews, removed reviews <3 words

### 4. Text Data Quality
- **Issue:** Emojis, URLs, special characters, very short/meaningless reviews
- **Solution:** Regex cleaning, emoji/URL removal, NLTK lemmatization and stopword removal, minimum word count filtering

### 5. Computational Resources
- **Issue:** Google Colab free tier session timeouts during long-running analysis
- **Solution:** Implemented checkpoint systems to save intermediate results, optimized code to reduce memory usage

---

## 📝 Changes from Original Proposal (PPS)

### Major Scope Adjustments

1. **From Continuous System to One-Time Analysis**
   - **Original:** Build recurring monthly analytics system
   - **Revised:** Single comprehensive analysis focused on 2027 data
   - **Reason:** Time constraints and lack of real-time hotel data integration

2. **From Monthly to Yearly Playbook**
   - **Original:** Generate monthly playbooks for continuous improvement
   - **Revised:** Consolidated yearly playbook encompassing all 2027 months
   - **Reason:** More manageable and practical for hotel management adoption

3. **Simplified Methodology**
   - **Original:** Multi-tool approach (TextBlob, NLTK, R tidytext)
   - **Revised:** Python-only (TextBlob, VADER, scikit-learn LDA)
   - **Reason:** Accessibility, reproducibility, faster implementation

4. **Data Source Reduction**
   - **Original:** Booking.com, TripAdvisor, and Agoda
   - **Revised:** Booking.com and TripAdvisor only (excluded Agoda)
   - **Reason:** Agoda lacked critical reviewer location data needed for geographic analysis

---

## 🤝 Stakeholder Analysis

### Core Stakeholders
- Project Lead (M.M. Senul Laksindu Semal)
- Project Supervisor (Mrs. Sapna Dissanayake)
- Cinnamon Hotels & Resorts Senior Management
- Hotel General Managers (12 Properties)

### Secondary Stakeholders
- Department Heads (Housekeeping, Front Desk, F&B, Facilities)
- Support Teams (Revenue Management, Marketing, Quality Assurance, IT, HR)
- Parent Company (John Keells Holdings)

### Tertiary Stakeholders
- Hotel Guests (Past & Future)
- Sri Lanka Tourism Development Authority (SLTDA)
- Competing Hotels in Sri Lanka
- Online Platforms (Booking.com, TripAdvisor)
- University of Westminster Academic Assessors

---


## 📄 License

This project is submitted as part of academic requirements for the BSc (Hons) Data Science and Analytics programme at the University of Westminster. All rights reserved.


---

## 📧 Contact Information

**Student:** M.M. Senul Laksindu Semal  
**Student ID:** w2000706  
**Email:** w2000706@my.westminster.ac.uk  
**Institution:** Informatics Institute of Technology (Affiliated with University of Westminster)  
**Programme:** BSc (Hons) Data Science and Analytics  

**Supervisor:** Mrs. Sapna Dissanayake  
**Department:** School of Computer Science & Engineering  

---

## 🙏 Acknowledgments

- **Cinnamon Hotels & Resorts** for project inspiration and domain context
- **Sri Lanka Tourism Development Authority (SLTDA)** for providing arrival statistics
- **Booking.com and TripAdvisor** for review data (accessed via Apify)
- **Mrs. Sapna Dissanayake** for supervision, guidance, and continuous support
- **University of Westminster** for academic resources and infrastructure
- **Informatics Institute of Technology (IIT)** for facilitating the programme
- **Apify** for reliable web scraping infrastructure
- Open-source community for Python libraries (NLTK, scikit-learn, pandas, etc.)

---

## 📌 Project Status

**Current Phase:** Advanced Analytics & Dashboard Development  
**Completion:** ~70%  
**Expected Final Submission:** April 2026  

**Last Updated:** February 2026

---

