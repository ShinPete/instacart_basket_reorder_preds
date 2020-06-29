# Instacart Market Basket Prediction Challenge
![Fraud_img](https://www.supermarketnews.com/sites/supermarketnews.com/files/styles/article_featured_standard/public/Instacart-Personal_Shopper-Bag.png?itok=LIpaSQ2n)

## Notebook
[My Notebook](https://nbviewer.jupyter.org/github/ShinPete/instacart_basket_reorder_preds/blob/master/Instacart_Reorder_preds.ipynb)

## Abstract:
We will be examining the Instacart Database and trying to acquire some valuable business insights out of this. And we will be trying answer three different questions relating to this database.

Question 1: What kind of products are purchased and reordered the most? The importance of this question is that we get a sense of what the most popular products are as well as maybe gaining an understanding of the multiple skews that are at play. Furthermore, we can decipher which departments are individually doing the best and we can opt to direct more support to flagging departments as a result.

Question 2: What factors influence one's decision to reorder a good and can we make a model that accurately predicts whether a good will be reordered? This is important because with this knowledge we can prepare our inventory based on the likleihood of a product being reordered in the future.Ideally we can shift our demand expectations to be closer in line with reality in general with the implementation of this.

## Methodology 
O —  Obtaining our data:
The Data was obtained from the Kaggle instacart dataset.

S — Scrubbing / Cleaning our data:
The data is largely already pretty clean and required little modification before it was ready to use outside the box.

E — Exploring / Visualizing our data will allow us to find patterns and trends:
Data Exploration Yeilded many very interesting insights. We very quickly learn that a tremendous amount of instacart's sales are from organic products. Considering that organic products tend to be pricier than the inorganic variety. We may consider the organic product as a large driver of revenues. We also get to see which departments are performing best and worst. We discover that produce, dairy, snacks, and beverages make up the contents of over 50 percent of all instacart orders. Furthermore, we discover that people tend to reorder more on weekly or monthly intervals.

M — Modeling our data will give us our predictive power as a wizard:
We produce a model that helps us predict whether a certain good will be reordered or not. This is really useful because it helps us anticipate future demand. Another model we make helps us see what customers similar to this customer purchase. This can assist us in determining what the complementary goods of a certain product is. If the demand of X and Y covary together we can see what a decrease in the price of X may increase the demand of Y as well.

N — Interpreting our data:
Our model has a 71% accuracy at predicting whether someone will or will not reorder a product. Our reccommendation system showed that whenever one is purchasing any kind of food product, it is probably good to reccommend to the consumer to purchase fresh fruit. Additionally, we got insights about very specific goods like what to reccommend when your customer purchases red wine (reccommend white wine!)

Conclusion: When looking for goods to reccommend in the very same basket, complimentary goods are a great choice. 
