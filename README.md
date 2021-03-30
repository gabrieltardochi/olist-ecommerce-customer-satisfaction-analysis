# Brazilian E-Commerce Public Dataset by Olist
A customer satisfaction analysis on brazilian e-commerce data using DeepLearning and visualization tools. Project steps:
1. Understand given datasets on [data/raw](https://github.com/ga-tardochisalles/olist-ecommerce-dataset/tree/main/data/raw), done in [src code/first_touch_eda.ipynb](https://github.com/ga-tardochisalles/olist-ecommerce-dataset/blob/main/src%20code/first_touch_eda.ipynb);
2. Preprocess data for sentiment analysis training, done in [src code/preprocess_sent_analysis.ipynb](https://github.com/ga-tardochisalles/olist-ecommerce-dataset/blob/main/src%20code/preprocess_sent_analysis.ipynb);
3. Fine tune BERTimbau(credits to NeuralMind) (BERT for ptbr text) on the sentiment classification task(~90% accuracy), done in [src code/train.py](https://github.com/ga-tardochisalles/olist-ecommerce-dataset/blob/main/src%20code/train.py)(I trained it using a [Kaggle kernel](https://www.kaggle.com/gabrieltardochi/sentiment-analysis-bertimbau-pt-br-order-reviews/output)), classify positivism on e-commerce order reviews;
4. Create a Dashboard focused on Customer Satisfaction indicators and analysis, done in [dashboard/Customer Satisfaction.pbix](https://github.com/ga-tardochisalles/olist-ecommerce-dataset/blob/main/dashboard/Customer%20Satisfaction.pbix).  

My LinkedIn: https://www.linkedin.com/in/gabriel-tardochi-salles-a1653a193/  
GitHub Portfolio: https://github.com/ga-tardochisalles  
My Medium Page: https://ga-tardochisalles.medium.com/
