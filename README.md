# 🛡️ Aftershock Nowcaster: Hybrid Geophysical-Econometric System

![GitHub release (latest by date)](https://img.shields.io/badge/Release-v1.0.0-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Ready-FF4B4B)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Following a catastrophic earthquake, the sequence of aftershocks often poses a greater threat to damaged infrastructure and rescue operations than the main event itself. **Aftershock Nowcaster** is a highly practical dashboard that produces a short-term risk forecast ("nowcast") by combining canonical geophysics laws with financial econometric volatility models.

---

## 📑 Table of Contents
1. [Dashboard Previews](#-dashboard-previews)
2. [Project Overview](#-project-overview)
3. [Methodological Flow](#-methodological-flow)
4. [Complete Findings](#-complete-findings)
5. [Installation & Usage](#-installation--usage)
6. [Scope & Future Work](#-scope--future-work)
7. [Directory Structure](#-directory-structure)

---

## 📸 Dashboard Previews

### 1. Live Spatial Mapping (Built-in Dataset)
![Live Mapping](images/earthquake_plotted.png)
*Interactive 3D heatmap rendering the physical locations and magnitudes of all recorded aftershocks using PyDeck and CartoDB Dark Matter.*

### 2. Geophysics Baseline (Omori-Utsu Decay)
![Omori Law Fit](images/omori_law.png)