# Used Car Price Prediction Dataset - Comprehensive Description

## Dataset Overview
This dataset contains comprehensive information about used cars listed for sale, with the primary objective of predicting the sale price of vehicles based on various features and characteristics.

---

## Dataset Metadata

### Dataset Size
- **Total Records:** 7,402 cars
- **Total Features:** 29 columns
- **File Format:** CSV (Comma-Separated Values)
- **File Name:** `Used_Car_Price_Prediction.csv`
- **File Size:** ~7.4K rows

---

## Features Description

### Target Variable
**`sale_price`** (Numeric - Continuous)
- **Description:** The selling price of the used car in Indian Rupees (INR)
- **Data Type:** Float/Integer
- **Range:** Approximately ₹164,899 to ₹680,558+
- **Purpose:** Primary prediction target in machine learning models

---

## Feature Categories

### 1. **Vehicle Information**
| Feature | Data Type | Description | Example Values |
|---------|-----------|-------------|-----------------|
| `car_name` | Text | Full name of the car model | "maruti swift", "hyundai grand i10" |
| `make` | Text | Car manufacturer/brand | maruti, hyundai, honda, ford, renault |
| `model` | Text | Specific model of the car | swift, alto 800, grand i10, verna |
| `variant` | Text | Specific variant with features | "lxi opt", "sports 1.2 vtvt", "vdi abs" |

### 2. **Vehicle Specifications**
| Feature | Data Type | Description | Example Values |
|---------|-----------|-------------|-----------------|
| `yr_mfr` | Integer | Year of manufacture | 2008-2018 |
| `body_type` | Text | Type of vehicle body | hatchback, sedan, suv |
| `fuel_type` | Text | Type of fuel used | petrol, diesel, petrol & cng |
| `transmission` | Text | Type of transmission | manual, automatic |
| `kms_run` | Integer | Kilometers already run by the car | 0-139,038+ km |

### 3. **Vehicle Condition & History**
| Feature | Data Type | Description | Example Values |
|---------|-----------|-------------|-----------------|
| `total_owners` | Integer | Number of previous owners | 1, 2, 3+ |
| `car_rating` | Text | Qualitative rating of the car | "great", "good", "average" |
| `fitness_certificate` | Boolean | Whether fitness certificate is available | True, False |
| `warranty_avail` | Boolean | Whether warranty is available | True, False |
| `reserved` | Boolean | Whether the car is reserved | True, False |
| `assured_buy` | Boolean | Whether assured buy option is available | True, False |

### 4. **Pricing Information**
| Feature | Data Type | Description | Example Values |
|---------|-----------|-------------|-----------------|
| `original_price` | Float | Original/manufacturer price | ₹154,164 to ₹703,554 |
| `broker_quote` | Float | Broker's quoted price (may have missing values) | Varies |
| `emi_starts_from` | Integer | Minimum EMI starting amount | ₹3,800 - ₹12,600+ |
| `booking_down_pymnt` | Integer | Down payment amount required | ₹24,735 - ₹81,405 |

### 5. **Location Information**
| Feature | Data Type | Description | Example Values |
|---------|-----------|-------------|-----------------|
| `city` | Text | City where car is listed | noida, delhi, gurgaon, faridabad |
| `registered_city` | Text | City where car is registered | delhi, noida, new delhi, agra |
| `registered_state` | Text | State where car is registered | delhi, uttar pradesh, haryana |
| `rto` | Text | RTO (Regional Transport Office) code | dl6c, up16, hr26 |

### 6. **Market & Listing Information**
| Feature | Data Type | Description | Example Values |
|---------|-----------|-------------|-----------------|
| `times_viewed` | Integer | Number of times the listing was viewed | 400-18,715+ views |
| `is_hot` | Boolean | Whether the listing is marked as "hot" | True, False |
| `source` | Text | Source of the listing | inperson_sale |
| `car_availability` | Text | Current availability status | in_stock, in_transit |
| `ad_created_on` | DateTime | Date and time when ad was created | 2020-11-07 to 2021-04-13 |

---

## Data Quality & Characteristics

### Missing Values
- Some features contain missing values (denoted as empty or NaN):
  - `broker_quote`: Approximately 10-15% missing
  - `transmission`: A few missing values
  - `original_price`: Sparse missing values

### Data Distribution
- **Categorical Features:** 12 (car_name, make, model, body_type, fuel_type, etc.)
- **Numerical Features:** 9 (yr_mfr, kms_run, sale_price, times_viewed, etc.)
- **Boolean Features:** 6 (assured_buy, fitness_certificate, etc.)
- **DateTime Features:** 1 (ad_created_on)
- **Text Features:** 5 (car_name, make, model, variant, city, etc.)

### Key Statistics on Target Variable (sale_price)
- **Mean Sale Price:** ~₹315,000 (approximate)
- **Median Sale Price:** ~₹287,000 (approximate)
- **Price Range:** ₹164,899 to ₹680,558+
- **Distribution:** Right-skewed (higher-priced luxury cars are less common)

---

## Feature Importance Indicators

### High Impact Features (Expected)
1. **Year of Manufacture (yr_mfr):** Newer cars typically cost more
2. **Kilometers Run (kms_run):** Higher mileage typically reduces price
3. **Make/Model:** Brand reputation significantly affects price
4. **Body Type:** Different body types have different price ranges
5. **Fuel Type:** Diesel cars generally cost more than petrol
6. **Original Price:** Baseline for estimating sale price

### Moderate Impact Features
1. **Times Viewed:** Popular listings may indicate good value
2. **Total Owners:** More owners may indicate wear and tear
3. **EMI Amount:** Correlates with vehicle price
4. **Location:** Different regions have different car prices

### Low-to-Moderate Impact Features
1. **Fitness Certificate:** Indicates proper maintenance
2. **Warranty Availability:** May affect buyer confidence
3. **Car Rating:** Qualitative assessment of condition
4. **RTO Code:** Geographical indicator

---

## Use Cases & Applications

### Machine Learning Tasks
1. **Regression:** Predict exact sale price
2. **Classification:** Categorize cars into price brackets (Low, Medium, High)
3. **Clustering:** Group similar cars for market analysis
4. **Feature Engineering:** Create derived features (car age, price depreciation %)

### Business Analytics
- Price trend analysis across regions
- Market demand assessment
- Competitive pricing strategy
- Inventory optimization

---

## Data Preparation Considerations

### Recommended Preprocessing Steps
1. **Handle Missing Values:**
   - Fill `broker_quote` with median or mean
   - Drop rows with critical missing data

2. **Feature Engineering:**
   - Create `car_age` from `yr_mfr`
   - Calculate `price_depreciation` (original_price - sale_price)
   - Extract `month` and `year` from `ad_created_on`

3. **Encoding:**
   - One-hot encode categorical features (fuel_type, body_type, etc.)
   - Label encode ordinal features if applicable
   - Map `car_rating` to numeric values (great→5, good→4, etc.)

4. **Normalization/Scaling:**
   - Normalize/scale numeric features (kms_run, times_viewed)
   - Min-Max or Standard Scaling recommended

5. **Outlier Detection:**
   - Identify unusually priced cars (price anomalies)
   - Check for data entry errors

---

## Data Collection Details

### Time Period
- **Collection Date Range:** November 2020 to April 2021
- **Data Currency:** Approximately 3-4 years old (as of 2024)

### Geography
- **Primary Region:** NCR Region (Delhi, NCR, Uttar Pradesh, Haryana)
- **Main Cities:** Noida, Delhi, Gurgaon, Faridabad

### Platforms
- **Source:** Personal sales (inperson_sale)
- **Market:** Used car marketplace/auction platform

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Records | 7,402 |
| Features | 29 |
| Target Variable | sale_price |
| Missing Data Percentage | ~5-15% (varies by feature) |
| Time Span | Nov 2020 - Apr 2021 |
| Geographic Scope | NCR Region, India |
| Primary Use Case | Price Prediction |

---

## References & Notes

1. **Data Source:** Used car listings from online marketplace
2. **Target Audience:** ML practitioners, data scientists, automotive analysts
3. **Predictive Power:** High (multiple relevant features available)
4. **Challenges:** Missing values, outliers, geographic bias, temporal changes

---

**Dataset Last Updated:** January 2026
**Documentation Version:** 1.0
