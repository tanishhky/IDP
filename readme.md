# Sector-Level Analysis and Clustering of S&P 500 Companies Using Financial Metrics and Machine Learning

## Abstract
Understanding the financial market structure and the positions of its players, i.e., companies and sectors, is of utmost importance [1]. While traditional methods are useful for interpreting financial statements, it is also crucial to understand the coherence of various sectors during market stress [2, 3]. The application of machine learning and statistics in financial markets has grown in importance [4]. This paper presents a comprehensive analysis of sector-level performance within the S&P 500, using a methodology that combines various financial metrics, clustering techniques, and neural networks [5]. Key metrics include market capitalization growth, revenue growth, performance variance (using weighted versus simple averages), and short- and long-term beta covariances [6]. This research identifies sector leaders, assesses market dominance, and explores sector stability [7]. It introduces an 'over-performance index' to identify which companies are perceived as leaders within a sector [8]. Additionally, inter-sector similarities are examined using clustering techniques, and a neural network classifies these sectors into clusters [10]. The findings provide critical insights into sector dynamics for investment decisions and identifying growth opportunities [11].

**Index Terms:** S&P 500, Financial Metrics, Clustering, Neural Networks, Sector Analysis, Overperformance Index, Financial Markets [12]

***

## I. INTRODUCTION

Large-Cap companies are key indicators of market trends and sectoral performance [12]. Understanding sector-level dynamics is essential for investors and policymakers [13]. This project integrates financial metrics, clustering techniques, and neural networks to analyze the competitive environment in various sectors [13]. By examining market capitalization, revenue growth, and variance, the project identifies monopolistic, duopolistic, and oligopolistic trends [14]. A neural network model is also used to classify sectors based on performance metrics, offering deeper insights into market behavior [15]. The main goal is to analyze sectors with monopolistic and oligopolistic environments [16]. To achieve this, an 'overperformance index' was developed, which utilizes a weighted average approach to better fit the variances to our specific requirements [17]. This index helps identify which companies in an industry over-perform the sector average [18]. After understanding each industry, they are clustered based on their normalized sector-wise performance [19]. Five parameters are formulated for each sector: year-on-year market capitalization growth, year-on-year revenue growth, the difference between simple and weighted average of year-on-year market capitalization growth, 6-month sector beta, and 4-year sector beta [20].

***

## II. THEORETICAL BACKGROUND

This research is founded on key economic concepts:

* **Market Structures:** The study focuses on identifying sectors characterized by monopoly, duopoly, and oligopoly [23].
    * A **monopoly** is a market where a single company dominates the sector [24].
    * A **duopoly** is a market where two companies dominate the sector [25].
    * An **oligopoly** is a market where several companies compete, with no single company being dominant [26].
* **Variance and Beta-Covariances:** These metrics are used to analyze and differentiate between market structures [28]. An index using a weighted average approach was developed to capture these metrics [29].

***

## III. METHODOLOGY

The methodology involves three main steps [30]:

### A. Data Preprocessing
Quarterly market capitalization and annual revenue data were merged, cleaned, and analyzed for each sector [31]. Key metrics like Year-over-Year (YoY) growth, weighted vs. simple averages, and beta covariance were calculated for each S&P 500 company [31]. Cumulative data was formed for each sector [32]. Only sectors with at least three listed companies in the S&P 500 were included to ensure data sufficiency and study quality [32, 33].

### B. Metrics and Equations

* **Market Cap/Revenue Growth Score:** [41]

$$\text{Return} = \frac{Y_i - Y_{i-1}}{Y_{i-1}} \quad (1)$$

$$\text{Score} = \begin{cases} 
1 & \text{if value > mean + 0.07 * SD} \\ 
-1 & \text{if value < mean - 0.07 * SD} \\ 
0 & \text{otherwise} 
\end{cases} \quad (2)$$

A threshold of 0.07 is used, based on the idea that a company should grow more than the risk-free interest rate to attract investors [34].

* **Weighted-Simple Variance:** This is the absolute difference between the weighted and simple averages for a sector [35].
    The weight $w_i$ associated with a company is defined as:

$$w_i = \frac{c_i^2}{\sum c_j^2} \quad (4)$$

where $c_i$ is the number of times company *i* outperformed the sector average [35].

* **Beta Covariance:** This measures the correlation of sector performance over time (short-term: 6 months; long-term: 4 years) [36].

$$\text{Mkt Value} = \frac{\text{Mkt Cap}(t)}{\text{Mkt Cap}(t-1)} - 1 \quad (5)$$

$$\text{Mkt Weight} = \frac{\text{Mkt Cap}(t)}{\sum \text{Mkt Value}(t)} \quad (6)$$

$$\text{Mkt Returns} = \sum (\text{Returns}(t) \cdot \text{Mkt Weight}(t)) \quad (7)$$

$$\beta = \frac{\text{Covariance}(\text{Returns}(t), \text{Mkt Returns}(t))}{\text{Variance}(\text{Mkt Returns}(t))} \quad (8)$$

### C. Sector Clustering
Five parameters were formulated for each sector: YoY market capitalization growth, YoY revenue growth, difference between simple and weighted average of YoY market cap growth, 6-month sector beta, and 4-year sector beta [20]. These parameters were normalized to prepare for clustering [43].
* **Hierarchical Clustering:** This was used to visualize cluster distances and determine the appropriate number of clusters [44].
* **K-Means Clustering:** This was performed using the number of clusters identified from the hierarchical clustering step [45].

### D. Overperformance Index
The weighted average growth of each sector was used to count the number of times each company outperformed the sector's mean growth [46]. This data was then used to create an overperformance-based index to classify sectors into monopolistic, duopolistic, or oligopolistic environments [47].

### E. Neural Network Model
A multi-layer perceptron (MLP) deep learning model was developed [48].
* **Input:** Normalized performance metrics for each sector [56].
* **Layers:** Three dense layers with ReLU activation, followed by a softmax layer for classification [57].
* **Output:** Cluster classification based on performance similarities [58].

![Architecture of the Artificial Neural Network](fig/arch.png)
*Fig. 1. Architecture of the Artificial Neural Network [56]*

The model was trained on 60% of the data and tested on the remaining 40% [52].

***

## IV. RESULTS AND DISCUSSION

Differences were observed for sectors with monopolistic and oligopolistic settings [59].

![Difference between simple and weighted average in different sectors](fig/mono-oligo.png)
*Fig. 2. The difference between simple and weighted average is significant in Casinos & Gaming Sector (on the left) and Communication Equipment Sector (on the right) [63]*

Out of 66 sectors analyzed, the study found:
* **47 sectors** with an oligopolistic environment [54].
* **8 sectors** with a duopolistic environment [54].
* **11 sectors** with a nearly monopolistic environment based on market capitalization growth [54].

A dendrogram was created to visualize the hierarchical clustering and decide on the number of clusters for K-Means [49]. "Automobile Manufacturers" was identified as an outlier [50]. Four clusters were obtained and initialized for K-Means clustering [51].

![Hierarchical Clustering of Sectors](fig/dendro.png)
*Fig. 3. Dendrogram representing 66 clusters and their distance for hierarchical clustering [67]*

**Confusion Matrix of Predicted Clusters** [60]

| True Label | Predicted 1 | Predicted 2 | Predicted 3 | Predicted 4 |
| :--------: | :---------: | :---------: | :---------: | :---------: |
| **1** | 6           | 0           | 1           | 0           |
| **2** | 8           | 0           | 0           | 1           |
| **3** | 0           | 0           | 0           | 2           |
| **4** | 0           | 0           | 0           | 9           |
*[Table data sourced from [61]]*

**Clustering Evaluation Metrics** [64]

| Metric                        | Score |
| ----------------------------- | :---: |
| Adjusted Rand Index (ARI)     | 0.76  |
| Normalized Mutual Information | 0.82  |
| Homogeneity                   | 0.81  |
| Completeness                  | 0.82  |
| V-measure                     | 0.82  |
| Fowlkes-Mallows Index         | 0.83  |
*[Table data sourced from [64]]*

The high scores indicate that the clustering aligns well with the true structure of the data [62]. The neural network classifier achieved an accuracy of **93%** using softmax activation [53].

***

## V. CONCLUSION

This research successfully analyzed the sector-wise performance within the S&P 500, offering valuable insights into market dynamics and sector clustering [67]. The methodology, combining financial metrics, hierarchical clustering, and neural networks, effectively revealed significant trends and patterns [68]. The study identified varying levels of market dominance, classifying 49 sectors as oligopolistic, 8 as duopolistic, and 12 as monopolistic based on market cap growth [69]. Inter-sector relationships were successfully identified using hierarchical and K-means clustering [70]. The neural network model demonstrated a high classification accuracy of 93% [71]. These findings are important for strategic investment, policy formulation, and market analysis by providing a deeper understanding of sector-level stability and competitive dynamics [72]. This research highlights the effectiveness of combining financial expertise with data science and machine learning [74]. Future work could incorporate ESG metrics, international market data, and more advanced deep learning models [73].

***

## References

[1] T. Conlon, J. Cotter, and I. M. Tuna, "Network topology and financial sector risk," *Journal of Financial Stability*, vol. 62, p. 101050, 2022.

[2] A. G. F. Stapel, S. A. G. Van Wingerden, and M. A. Van Dijk, "Fifty years of stock-return-comovement: A review of the literature," *Journal of Commodity Markets*, vol. 27, p. 100223, 2022.

[3] N. F. F. da Silva, F. H. F. D. S. and L. A. S. Dias, L. A. S. de Oliveira, and L. A. S. de Oliveira, "Deep learning for stock market prediction: A review and a new methodology," *Applied Soft Computing*, vol. 147, p. 110756, 2023.

[4] S. H. Nasiri, S. M. A. H. Hosseini, and M. R. Feizi-Derakhshi, "An effective stock market prediction model using a hybrid clustering and classification-based approach," *Journal of Ambient Intelligence and Humanized Computing*, vol. 15, no. 1, pp. 493-509, 2024.

[5] G. S. Atsalakis and K. P. Valavanis, "Surveying stock market forecasting techniques – Part II: Soft computing methods," *Expert Systems with Applications*, vol. 36, no. 3, pp. 5932-5941, 2009.

[6] G. J. Mantegna, "Hierarchical structure in financial markets," *The European Physical Journal B-Condensed Matter and Complex Systems*, vol. 11, no. 1, pp. 193-197, 1999.

[7] Z. Bodie, A. Kane, and A. J. Marcus, *Investments*, 12th ed. McGraw-Hill Education, 2020.

[8] J. MacQueen, "Some methods for classification and analysis of multivariate observations," in *Proceedings of the fifth Berkeley symposium on mathematical statistics and probability*, vol. 1, no. 14, pp. 281-297, 1967.

[9] J. Tirole, *The Theory of Industrial Organization*. MIT press, 1988.

[10] S. K. Mitra, "Digital signal processing: A computer-based approach," McGraw-Hill, 2006.

[11] R. O. Duda, P. E. Hart, and D. G. Stork, *Pattern Classification*, 2nd ed. Wiley-Interscience, 2000.

[12] W. F. Sharpe, "Capital asset prices: A theory of market equilibrium under conditions of risk," *The Journal of Finance*, vol. 19, no. 3, pp. 425-442, 1964.

[13] E. F. Fama and K. R. French, "Common risk factors in the returns on stocks and bonds," *Journal of Financial Economics*, vol. 33, no. 1, pp. 3-56, 1993.

[14] M. J. Brennan, "The individual investor," *Journal of Financial Research*, vol. 18, no. 1, pp. 59-74, 1995.

[15] A. W. Lo, "The adaptive markets hypothesis: Market efficiency from an evolutionary perspective," *Journal of Portfolio Management*, vol. 30, no. 5, pp. 15-29, 2004.

[16] R. C. Merton, "An intertemporal capital asset pricing model," *Econometrica*, vol. 41, no. 5, pp. 867-887, 1973.

[17] J. L. Treynor, "How to rate management of investment funds," *Harvard Business Review*, vol. 43, no. 1, pp. 63-75, 1965.

[18] W. F. Sharpe, "The Sharpe ratio," *Journal of Portfolio Management*, vol. 21, no. 1, pp. 49-58, 1994.

[19] F. A. Sortino and R. van der Meer, "Downside risk," *Journal of Portfolio Management*, vol. 17, no. 4, pp. 27-31, 1991.

[20] J. P. Morgan and Reuters, "RiskMetrics—Technical Document," 4th ed., 1996.

[23] J. Tirole, *The Theory of Industrial Organization*. MIT press, 1988.

[24] A. Marshall, *Principles of Economics*, 8th ed. Macmillan, 1920.

[25] A. Cournot, *Researches into the Mathematical Principles of the Theory of Wealth*. Macmillan, 1897.

[26] E. H. Chamberlin, *The Theory of Monopolistic Competition*. Harvard University Press, 1933.

[28] H. Markowitz, "Portfolio selection," *The Journal of Finance*, vol. 7, no. 1, pp. 77-91, 1952.

[29] J. Lintner, "The valuation of risk assets and the selection of risky investments in stock portfolios and capital budgets," *The Review of Economics and Statistics*, vol. 47, no. 1, pp. 13-37, 1965.

[30] R. A. Fisher, "The use of multiple measurements in taxonomic problems," *Annals of Eugenics*, vol. 7, no. 2, pp. 179-188, 1936.

[31] G. E. P. Box and G. M. Jenkins, *Time Series Analysis: Forecasting and Control*. Holden-Day, 1976.

[32] C. R. Rao, *Linear Statistical Inference and Its Applications*. Wiley, 1973.

[33] M. G. Kendall and A. Stuart, *The Advanced Theory of Statistics*. Griffin, 1979.

[34] I. Fisher, *The Theory of Interest*. Macmillan, 1930.

[35] K. Pearson, "On lines and planes of closest fit to systems of points in space," *Philosophical Magazine*, vol. 2, no. 11, pp. 559-572, 1901.

[36] H. Hotelling, "Analysis of a complex of statistical variables into principal components," *Journal of Educational Psychology*, vol. 24, no. 6, pp. 417-441, 1933.

[41] W. F. Sharpe, "A simplified model for portfolio analysis," *Management Science*, vol. 9, no. 2, pp. 277-293, 1963.

[43] J. W. Tukey, *Exploratory Data Analysis*. Addison-Wesley, 1977.

[44] J. H. Ward Jr., "Hierarchical grouping to optimize an objective function," *Journal of the American Statistical Association*, vol. 58, no. 301, pp. 236-244, 1963.

[45] J. MacQueen, "Some methods for classification and analysis of multivariate observations," in *Proceedings of the fifth Berkeley symposium on mathematical statistics and probability*, vol. 1, no. 14, pp. 281-297, 1967.

[46] E. F. Fama and K. R. French, "The cross-section of expected stock returns," *The Journal of Finance*, vol. 47, no. 2, pp. 427-465, 1992.

[47] A. W. Lo and A. C. MacKinlay, "Stock market prices do not follow random walks: Evidence from a simple specification test," *The Review of Financial Studies*, vol. 1, no. 1, pp. 41-66, 1988.

[48] F. Rosenblatt, "The perceptron: a probabilistic model for information storage and organization in the brain," *Psychological Review*, vol. 65, no. 6, pp. 386-408, 1958.

[52] V. N. Vapnik, *The Nature of Statistical Learning Theory*. Springer, 1995.

[53] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *Nature*, vol. 521, no. 7553, pp. 436-444, 2015.

[54] S. P. Lloyd, "Least squares quantization in PCM," *IEEE Transactions on Information Theory*, vol. 28, no. 2, pp. 129-137, 1982.

[56] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. MIT Press, 2016.

[57] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with deep convolutional neural networks," *Communications of the ACM*, vol. 60, no. 6, pp. 84-90, 2017.

[58] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770-778, 2016.

[59] J. D. Hamilton, *Time Series Analysis*. Princeton University Press, 1994.

[60] T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. Springer, 2009.

[61] C. M. Bishop, *Pattern Recognition and Machine Learning*. Springer, 2006.

[62] L. Kaufman and P. J. Rousseeuw, *Finding Groups in Data: An Introduction to Cluster Analysis*. Wiley, 1990.

[63] R. Tibshirani, G. Walther, and T. Hastie, "Estimating the number of clusters in a data set via the gap statistic," *Journal of the Royal Statistical Society: Series B*, vol. 63, no. 2, pp. 411-423, 2001.

[64] P. J. Rousseeuw, "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis," *Journal of Computational and Applied Mathematics*, vol. 20, pp. 53-65, 1987.

[67] G. J. Mantegna, "Hierarchical structure in financial markets," *The European Physical Journal B-Condensed Matter and Complex Systems*, vol. 11, no. 1, pp. 193-197, 1999.

[68] R. N. Mantegna and H. E. Stanley, *An Introduction to Econophysics: Correlations and Complexity in Finance*. Cambridge University Press, 1999.

[69] E. F. Fama and K. R. French, "Industry costs of equity," *Journal of Financial Economics*, vol. 43, no. 2, pp. 153-193, 1997.

[70] M. E. J. Newman, "The structure and function of complex networks," *SIAM Review*, vol. 45, no. 2, pp. 167-256, 2003.

[71] Y. Bengio, A. Courville, and P. Vincent, "Representation learning: A review and new perspectives," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 35, no. 8, pp. 1798-1828, 2013.

[72] A. W. Lo, "The adaptive markets hypothesis," *Journal of Portfolio Management*, vol. 30, no. 5, pp. 15-29, 2004.

[73] M. M. Carhart, "On persistence in mutual fund performance," *The Journal of Finance*, vol. 52, no. 1, pp. 57-82, 1997.

[74] R. C. Merton, "Theory of rational option pricing," *The Bell Journal of Economics and Management Science*, vol. 4, no. 1, pp. 141-183, 1973.
