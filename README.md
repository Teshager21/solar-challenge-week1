# 🌞 Solar Challenge: Cross-Country Solar Farm Analysis

> **Challenge to Kickstart AI Mastery with Cross-Country Solar Farm Analysis**  
> A data-driven comparison of solar energy potential in Benin 🇧🇯, Sierra Leone 🇸🇱, and Togo 🇹🇬

---

An interactive dashboard to explore solar potential across Benin, Togo, and Sierra Leone.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://solar-challenge-week1-srciwinqcgpf22litkpyhb.streamlit.app/)



## 📸 Streamlit App Preview

![Streamlit App Screenshot](images/streamlit.png)
## 📌 Overview

This project was developed as part of the **10 Academy Week 1 Challenge**, aimed at assessing solar energy potential across three West African countries — **Benin**, **Sierra Leone**, and **Togo** — to support strategic investment decisions by **MoonLight Energy Solutions**.

We compare solar irradiance metrics and environmental conditions to evaluate each country's suitability for solar power deployment.

---

## 🧭 Background

🌞 **MoonLight Energy Solutions** is focused on scaling clean energy by identifying optimal regions for solar installations.

### Key Irradiance Metrics:

- **GHI (Global Horizontal Irradiance)** – Total solar radiation on a horizontal surface.
- **DNI (Direct Normal Irradiance)** – Solar radiation in line with the sun’s rays.
- **DHI (Diffuse Horizontal Irradiance)** – Scattered solar radiation received indirectly.

Additional variables analyzed:  
☁️ Air Temperature • 💧 Humidity • 🌬️ Wind Speed • 🌧️ Precipitation • 🧽 Sensor Cleaning Events

---

## 🗂️ Project Structure

```
solar-challenge-week1/
├── data/                   # Cleaned and raw data
├── notebooks/              # Jupyter notebooks for analysis and visualization
├── scripts/                # Custom visualization and helper functions
├── tests/                  # Testing scripts
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/Teshager21/solar-challenge-week1.git
cd solar-challenge-week1
```

2. **Create and Activate Virtual Environment (Recommended)**

```bash
python -m venv venv
source venv/bin/activate       # On Linux/macOS
venv\Scripts\activate        # On Windows
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## 📊 Methodology

- 📥 Load and merge datasets from all three countries
- 🧼 Clean and preprocess data
- 📊 Visualize solar metrics using boxplots and bar charts
- 📈 Perform statistical tests (ANOVA & Kruskal-Wallis)
- 🧠 Extract insights and investment recommendations

---

## 🔍 Key Insights

### 💡 GHI (Global Horizontal Irradiance)
- **Togo** has the highest median and peak GHI.
- **Benin** and **Sierra Leone** have similar but slightly lower GHI levels.

### 💡 DNI (Direct Normal Irradiance)
- **Togo** again leads with the highest DNI — favorable for CSP and sun-tracking PV.
- **Benin** and **Sierra Leone** show lower values.

### 💡 DHI (Diffuse Horizontal Irradiance)
- All three countries have similar DHI distributions — important for cloudy-day PV generation.

---

## ✅ Country Suitability for Solar Investment

| Country       | Summary                                                                 |
|---------------|-------------------------------------------------------------------------|
| 🇹🇬 **Togo**       | 🌟 Most promising. Strong GHI & DNI. Suitable for fixed & tracking systems. |
| 🇧🇯 **Benin**      | 👍 Good potential. Slightly lower DNI, but competitive GHI.              |
| 🇸🇱 **Sierra Leone** | ⚠️ Feasible, but lower solar intensity. May require efficient technologies. |

---

## 📊 Statistical Tests (GHI)

- **ANOVA F-statistic:** 163.54  
- **p-value:** 0.0000  
- **Kruskal–Wallis H-statistic:** 428.27  
- **p-value:** 0.0000

> These results confirm **statistically significant** differences in GHI between the countries.

⚠️ **Note**: The dataset includes **1.5 million+ records**, which increases statistical power — even small differences may yield low p-values. Practical significance should also be evaluated.

---

## 📌 Recommendations

- ✅ **Invest in Togo** for both traditional PV and CSP systems due to high GHI & DNI.
- ✅ **Explore Benin** as a strong secondary location with competitive solar potential.
- ⚠️ **Approach Sierra Leone strategically**, focusing on hybrid or high-efficiency installations to overcome slightly lower irradiance levels.

---

## 📈 Visual Outputs

- 📊 **Boxplots**: GHI, DNI, DHI distributions by country
- 📉 **Bar Charts**: Country-level average GHI
- 📋 **Summary Table**: Mean, Median, and Std for GHI, DNI, and DHI

---

## 🧪 Run Tests

To run unit tests:

```bash
python -m unittest discover tests
```

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- ⭐ Star the repo
- 📂 Fork the project
- 🐛 Open issues
- 📥 Submit pull requests

---

## 📬 Contact

- **Author:** Teshager Admasu
- **GitHub:** [@Teshager21](https://github.com/Teshager21)
- **LinkedIn:** [@Teshager Admasu](https://www.linkedin.com/in/teshager-admasu-531090191)
- **Gmail:** [@Teshager](mailto:teshager8922@gmail.com)

---

> *Empowering sustainable energy solutions through data-driven insights.*