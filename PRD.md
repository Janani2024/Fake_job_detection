
📄 Product Requirements Document (PRD)

Hybrid AI-Based Fake Job Posting Detection System

---

1. 📌 Product Overview

**Product Name:** Hybrid AI-Based Fake Job Posting Detection System
**Product Type:** AI-powered Web Application
**Platform:** Web (Streamlit-based)

🎯 Purpose

To detect and prevent fraudulent job postings using machine learning and explainable AI, helping job seekers make safe and informed decisions.

🚨 Problem Statement

Online job platforms have seen a surge in fake job postings that:

* Mislead candidates with unrealistic offers
* Collect sensitive personal data
* Demand illegal payments

This creates **financial loss, data breaches, and trust issues** among users.

💡 Solution

Develop an intelligent system that:

* Classifies job postings as **Real or Fake**
* Provides **explanations for predictions**
* Enables **real-time analysis via web interface**

---

2. 🎯 Goals & Objectives

Primary Goals

* Accurately detect fraudulent job postings
* Improve **recall for fake job detection**
* Provide **transparent AI explanations**
* Deliver a **user-friendly interface**

Success Metrics (KPIs)

* Model Accuracy ≥ 90%
* Recall (Fake Jobs) ≥ 85%
* Response Time ≤ 3 seconds
* User satisfaction (qualitative feedback)

---

3. 👥 Target Users

* 🧑‍💼 Job seekers (freshers & professionals)
* 🎓 Students searching for internships/jobs
* 🏢 Recruitment platforms
* 🔐 Cybersecurity analysts

---

4. 🧩 Features & Functional Requirements

4.1 Core Features

🔍 1. Job Input Methods

* Text input (manual job description)
* URL input (scraped job posting)

🤖 2. Fake Job Detection Engine

* Classifies job as:

  * ✅ Real
  * ❌ Fake

📊 3. Explainable AI (LIME)

* Highlights:

  * Suspicious words/phrases
  * Key contributing features

📈 4. Model Comparison (Optional UI Feature)

* Show performance of:

  * Logistic Regression
  * Naive Bayes
  * SVM
  * Random Forest

🌐 5. Web Interface (Streamlit)

* Simple UI
* Real-time prediction
* Interactive explanation display

---

### 4.2 Functional Requirements

| ID  | Requirement                              |
| --- | ---------------------------------------- |
| FR1 | System must accept job text input        |
| FR2 | System must extract job content from URL |
| FR3 | System must preprocess text data         |
| FR4 | System must convert text using TF-IDF    |
| FR5 | System must classify job postings        |
| FR6 | System must display prediction results   |
| FR7 | System must generate LIME explanations   |
| FR8 | System must handle imbalanced data       |
| FR9 | System must respond within 3 seconds     |

---

5. ⚙️ Technical Requirements

5.1 Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **ML Libraries:**

  * Scikit-learn
  * Pandas, NumPy
* Explainability:LIME
* Text Processing:TF-IDF

---

5.2 Machine Learning Pipeline

📌 Data Processing

* Remove stopwords
* Handle missing values
* Combine features:

  * Title
  * Description
  * Requirements
  * Benefits

📌 Feature Engineering

* TF-IDF Vectorization

📌 Models Used

* Logistic Regression
* Naive Bayes
* Support Vector Machine (Best Performer)
* Random Forest

📌 Optimization

* Class balancing (to handle imbalanced dataset)

---

## 6. 🧠 System Architecture

```
User Input → Preprocessing → TF-IDF → ML Model (SVM)
                                      ↓
                              Prediction (Real/Fake)
                                      ↓
                               LIME Explanation
                                      ↓
                                Streamlit UI
```

---

7. 🖥️ User Flow

1. User opens web application
2. Enters job description or URL
3. Clicks “Analyze”
4. System processes input
5. Displays:

   * Prediction (Real/Fake)
   * Confidence score
   * Explanation (highlighted keywords)

---

8. 🎨 UI/UX Requirements

* Clean and minimal design
* Input box + URL option
* “Analyze” button
* Result display:

  * Color-coded output (Green = Real, Red = Fake)
* Highlighted explanation text

---

9. 🔐 Non-Functional Requirements

Performance

* Fast inference (< 3 seconds)

Scalability

* Can handle multiple users simultaneously

Security

* No storage of user input data
* Safe URL handling

Usability

* Beginner-friendly interface

---

10. ⚠️ Risks & Mitigation

| Risk                           | Mitigation           |
| ------------------------------ | -------------------- |
| Imbalanced dataset             | Use class weights    |
| False negatives (missed scams) | Optimize recall      |
| Poor explanations              | Use LIME             |
| Noisy input data               | Strong preprocessing |

---

11. 📊 Future Enhancements

* 🔗 Integration with job portals (LinkedIn, Indeed)
* 📱 Mobile app version
* 🧠 Deep Learning models (BERT, NLP transformers)
* 🌍 Multi-language support
* 🚨 Scam reporting system
* 📡 Real-time API integration

---

12. 📅 Timeline (Suggested)

| Phase                | Duration |
| -------------------- | -------- |
| Data Collection      | 1 week   |
| Preprocessing        | 1 week   |
| Model Training       | 2 weeks  |
| Evaluation           | 1 week   |
| UI Development       | 1 week   |
| Testing & Deployment | 1 week   |

---

13. 📌 Conclusion

This product leverages **machine learning + explainable AI** to tackle a real-world problem—fake job postings. By combining **high detection accuracy with transparency**, it builds trust and empowers users to make safe career decisions.

