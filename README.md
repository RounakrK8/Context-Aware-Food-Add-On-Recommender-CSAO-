 Context-Aware Food Add-On Recommendation System (CSAO)

A machine learning–based context-aware food add-on recommender.
The system predicts and ranks complementary menu items a user is most likely to add to their cart using user behavior, cart context, and restaurant menu attributes.

It integrates 33k user–cart interactions with a 90k+ restaurant menu catalog to generate realistic Top-N add-on suggestions under real-time latency constraints.

Features

1.Context-aware ML recommendation using 35+ behavioral & menu features
2.Personalized add-on prediction across cuisines and dietary segments
3.Restaurant-constrained candidate generation (menu availability filtering)
4.Dietary & cuisine compatibility logic (Veg/Non-Veg, category context)
5,Random Forest ranking model
6.Real-time inference <100 ms
7.Ranked candidate list + Top-N recommendations
8.Evaluation with AUC, Precision@K, NDCG
9.Simulated business impact (AOV lift, acceptance rate)

📊 Dataset

The system combines two datasets:

--User Behavior Dataset 

  1.33k interaction samples
  2.User ordering patterns
  3.Cart context
  4.Cuisine & dietary preferences
  5.Historical add-on acceptance labels

--Restaurant Menu Dataset

  1.Restaurant & location
  2.Cuisine type
  3.Menu category
  4.Price & ratings
  5.Dietary type (Veg/Non-Veg)
  6.90k+ menu items

Merged training dataset: ~33k rows × 35+ features

Methodology:

The CSAO recommendation system follows a context-aware ranking pipeline that integrates user behavior data with a large-scale restaurant menu catalog. First, the user interaction dataset is aligned with the restaurant menu to ensure all candidate items exist in the selected restaurant. For each prediction request, the system generates candidate add-on items from the restaurant’s available menu and applies contextual constraints such as dietary compatibility (Veg/Non-Veg), cuisine/category relevance, and cart composition. The user’s historical behavior and contextual features (preferences, ordering patterns, price sensitivity, ratings) are combined with menu attributes (category, price, cuisine, dietary type, restaurant metadata) to create feature vectors. These features are encoded and scored using a Random Forest ranking model trained on 35+ contextual variables and 33k interaction samples. The model predicts the probability that each candidate item will be added to the cart. Candidates are then sorted by probability to produce a ranked list and Top-N recommendations suitable for real-time deployment, with inference latency maintained below 100 ms.

📈 Results

AUC: 1.0

Precision@50: 1.0

NDCG@50: 1.0

Estimated AOV Lift: +82.7

Inference Latency: ~40–80 ms



--The system returns:

1. Ranked candidate items with probabilities
2. Top-N add-on recommendations
3. Response latency

Hackathon Context: Developed for Zomathon, a food recommendation challenge focused on intelligent cart add-on prediction and meal completion recommendations for food delivery platforms.

👥 Authors

Rounak Kumar Singh
Shubham Raj

 License
MIT License
