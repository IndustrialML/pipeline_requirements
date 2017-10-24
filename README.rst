===========================
Industrial Machine Learning
===========================
************************************************************
Requirements for productionizing a Machine Learning pipeline
************************************************************
*Joel Akeret, 2017*

----------------------------------------
1.	Productionizing Machine Learning
----------------------------------------
Operating an online machine learning system at large scale in a real-world application gives rise to a variety of challenges not present in research projects, toy examples or offline experiments. Contrary to software systems the challenges lay not only on the code level but rather on system level. Meaning that the overall performance of the machine learning system is influenced by the code and at the same time data quality. Properly addressing those challenges are key for a successful operation of such a system.

This document addresses recommendation regarding the design of a large machine learning system in a setup where features are extracted from raw input data, fed into a training system that produces a model, which is then read into a serving system to make inferences and affect the user-facing behavior of the system [1] [2].

Traditional software engineering practice has shown that strong abstraction boundaries using encapsulation and modular design help create maintainable code in which it is easy to make isolated changes and improvements. Strict abstraction boundaries help express the invariants and logical consistency of the information inputs and outputs from an given component. Unfortunately, it is difficult to enforce strict abstraction boundaries for machine learning systems by prescribing specific intended behavior. Machine learning systems mix signals together, entangling them and making isolation of improvements impossible [2]. 

In classical software engineering setting code dependencies can be identified via static code analysis by IDEs or compilers. Lacking similar tooling data dependencies can easily grow and become difficult to maintain and untangle.

--------------------------------------------
2.	Anatomy of a Machine Learning system
--------------------------------------------
In a full machine learning system only a small fraction of the code is actually centered around the algorithmic part, namely the training and inference of the model. The vast and complex surrounding infrastructure are concerned with the data processing and bringing the components together.

This being said, machine learning systems are more difficult to develop and maintain as they have to deal with the complexity and technical dept of the software system and additionally with the challenges arising through the machine learning components. 

Dividing a machine learning system into small and testable components and defining quality metric can mitigate the challenges. Figure 1 depicts the high-level anatomy of a machine learning system with its quality.

.. image:: https://raw.githubusercontent.com/IndustrialML/pipeline_requirements/master/imgs/components.png

*Figure 1 High-level overview of the different pipeline and quality components of a machine learning system*

2.1 Pipeline Components
^^^^^^^^^^^^^^^^^^^^^^^
Generally, a machine learning system consists of two phases: training and serving. Common to both is a preceding step, namely the preprocessing of the data being fed into the system. It is absolutely crucial that the same preprocessing is applied in both phase in order to ensure a fully functioning pipeline. Table 1 lists the different components of a regular machine learning system.

+------------------------+---------------------+---------------------------------------------------------------------------------------------+
|                        | **Component**       | **Description**                                                                             |
+------------------------+---------------------+---------------------------------------------------------------------------------------------+
| **Data preprocessing** | Data ingestion      | Components tasked to load and aggregate data from different systems.                        |
+------------------------+---------------------+---------------------------------------------------------------------------------------------+
|                        | Data transformation | Transforms and computes model features                                                      |
+------------------------+---------------------+---------------------------------------------------------------------------------------------+
| **Model serving**      | Model loading       | Loads the coefficients and hyperparameters from a serialized model                          |
+------------------------+---------------------+---------------------------------------------------------------------------------------------+
|                        | Model inference     | Calls the model with new data and post processes the inference result                       |
+------------------------+---------------------+---------------------------------------------------------------------------------------------+
| **Model training**     | Model trainer       | Responsible for the training of a specific model                                            |
+------------------------+---------------------+---------------------------------------------------------------------------------------------+
|                        | Model DB            | Store holding trained models that can either be used for the model evaluation or deployment |
+------------------------+---------------------+---------------------------------------------------------------------------------------------+

*Table 1 Machine learning pipeline components*

2.2 Quality components
^^^^^^^^^^^^^^^^^^^^^^
In order to be able to maintain a high level of quality within a machine learning system various different quality assessments have to be performed on a continuous basis on all the components. Additional to well established software best practices such as unit and integration testing the code further quality improving steps have to be implemented to ensure the quality of the data. 

The performance of a machine learning model crucially depends on the data being fed in. This is valid, both during training and inference time. Understanding the data and detecting anomalies early, prevent hard to identify bugs downstream the pipeline. As new versions of a model are being trained and deployed the performance of the model has to be assessed and compared with its preceding version.

As with regular software systems, machine learning systems should undergo rigorous automated tests. Broadly, the quality assessments can be grouped into the following metrics:

2.2.1	Data analysis
"""""""""""""""""""""
In order to ensure that incoming data is fulfilling basic requirements summary statistics should be computed over fresh data and then be compared to the statistics of the training data in order to detect changes or problems in data providers upstream.
 
2.2.2	Data validation
"""""""""""""""""""""""
After the data has undergone the basic analysis further check in from of data validation should be performed. Those validations should check if features are present in the data, if each feature has the appropriate type, if a minimum of examples are present in each feature and if each feature spans in the expected bounds. 

2.2.3	Model verification
""""""""""""""""""""""""""
This metric ensures that a newly trained model is save to be served to production. This checks basic requirements such as that the model can be loaded into the serving system and if it fulfils a minimal performance in a comparison to a baseline model.

2.2.4	Model evaluation
""""""""""""""""""""""""
By computing scores (e.g. AUC, F1, etc.) with the newly trained model on hold-out data the performance difference can be compared to previous models. This allows for decision making if the new model should be served to production.

2.2.5	Pipeline monitoring
"""""""""""""""""""""""""""
This metric serves the purpose to analyze the overall quality of the entire machine learning system. Ensuring steady performance of the model within the system. Alerting in cases where the unchanged system start to behave differently due to external factors.

2.3	Quality Testing
^^^^^^^^^^^^^^^^^^^^^^
Table 2 lists a set of high-level tests that should be performed by the different quality components on the machine learning system. The test are mostly derived from the ML Test Score [1] and should be implemented such that they can be executed automatically and repeatedly.


+-------------------------+----------------------------------------------------------------------------------------------------------------------+
| **Quality component**   | **Test**                                                                                                             |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+
| **Data analysis**       | Test that the distributions of each feature match your expectations.                                                 |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+
|                         | Test the relationship between each feature and the target, and the pairwise correlations between individual signals. |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+
|                         | Test for upstream instability in features, both in training and serving.                                             |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+
| **Data validation**     | Test all code that creates input features, both in training and serving.                                             |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+
|                         | Test that your training and serving features compute the same values.                                                |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+
|                         | Test for NaNs or infinities appearing in your model during training or serving.                                      |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+
| **Model evaluation**    | Test the relationship between offline proxy metrics and the actual impact metrics                                    |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+
|                         | Test the impact of each tunable hyperparameter.                                                                      |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+
|                         | Test the effect of model staleness.                                                                                  |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+
| **Model verification**  | Test against a simpler model as a baseline                                                                           |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+
|                         | Test the reproducibility of training.                                                                                |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+
|                         | Unit test model specification code.                                                                                  |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+
|                         | Integration test the full ML pipeline.                                                                               |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+
|                         | Test model quality before attempting to serve it.                                                                    |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+
|                         | Test models via a canary process before they enter production serving environments                                   |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+
|                         | Test how quickly and safely a model can be rolled back to a previous serving version.                                |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+
| **Pipeline monitoring** | Test for dramatic or slow-leak regressions in training speed, serving latency, throughput, or RAM usage.             |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+
|                         | Test for regressions in prediction quality on served data.                                                           |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+
|                         |                                                                                                                      |
+-------------------------+----------------------------------------------------------------------------------------------------------------------+

*Table 2 Quality ensuring tests grouped by the corresponding component*


---------------
3. Bibliography
---------------

[1] 
E. B. H.-T. C. N. F. C. Y. F. Z. H. S. H. M. I. V. J. L. K. C. Y. K. L. L. C. M. A. N. M. N. P. S. R. S. R. S. E. W. M. W. J. W. X. Z. M. Z. Denis Baylor, "TFX: A TensorFlow-Based Production-Scale Machine Learning Platform," KDD 2017 Applied Data Science Paper, 2017. 

[2] 
E. Brewck, S. Cai, E. Nielsen, M. Salib and D. Sculley, "What’s your ML Test Score? A rubric for ML production systems," Neural Information Processing Systems, 2016. 

[3] 
D. Scully, G. Holt, D. Golovin, E. Davydov, T. Philips, D. Ebner, V. Chaudhary, M. Young, J.-F. Crspo and D. Dennison, "Hidden technical debt in machine learning systems," Advances in Neural Information Processing Systems, 2015. 


