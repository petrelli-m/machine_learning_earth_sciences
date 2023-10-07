In [01]: my_grid_search.best_estimator_
Out[01]: SVC(C=10, kernel='linear')

In [02]: my_grid_search.best_score_
Out[02]: 0.9778761061946903

In [03]: my_grid_search.cv_results_
Out[03]: 
{'mean_fit_time': array([0.00605977, 0.02105349, 0.00482285, 
					     0.01113951, 0.00554657, 0.00662667]),
 'std_fit_time': array([3.7539e-04, 6.0314e-04, 2.1346e-04,
  				        7.0395e-04, 5.5384e-04, 3.1989e-05]),
 'mean_score_time': array([0.00242817, 0.01987976, 0.00181627,
 						   0.00979179, 0.00133586,0.00618142]),
 'std_score_time': array([7.4277e-05, 1.6316e-03, 1.6929e-04,
  						  2.7074e-04, 2.2063e-04, 6.4881e-04]),
 'param_C': masked_array(data=[0.1, 0.1, 1, 1, 10, 10],
              mask=[False, False, False, False, False, False],
              fill_value='?', dtype=object),
 'param_kernel': masked_array(data=['linear', 'rbf', 'linear', 
                                    'rbf',  'linear', 'rbf'],
              mask=[False, False, False, False, False, False],
              fill_value='?', dtype=object),
 'params': 	 [{'C': 0.1, 'kernel': 'linear'},
			  {'C': 0.1, 'kernel': 'rbf'},
			  {'C': 1, 'kernel': 'linear'},
			  {'C': 1, 'kernel': 'rbf'},
			  {'C': 10, 'kernel': 'linear'},
			  {'C': 10, 'kernel': 'rbf'}],
 'split0_test_score': array([0.92330383, 0.8879056 , 0.98230088,
 							 0.91150442, 0.97935103, 0.97050147]),
 'split1_test_score': array([0.9380531 , 0.88495575, 0.97935103,
  							 0.92625369, 0.98525074, 0.97935103]),
 'split2_test_score': array([0.92330383, 0.89380531, 0.97345133,
  							 0.91740413, 0.97640118, 0.96460177]),
 'split3_test_score': array([0.91740413, 0.88495575, 0.96755162,
  							 0.90560472, 0.97050147, 0.96460177]),
 'mean_test_score': array([0.92551622, 0.8879056 , 0.97566372,
 						   0.91519174, 0.97787611, 0.96976401]),
 'std_test_score': array([0.00762838, 0.00361282, 0.00566456, 
 						  0.00762838, 0.00531792, 0.0060364 ]),
 'rank_test_score': array([4, 6, 2, 5, 1, 3], dtype=int32)}