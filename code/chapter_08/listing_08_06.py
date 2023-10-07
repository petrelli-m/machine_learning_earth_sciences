def pre_processing_pipeline(input_files, out_file): 
    
    try:
        os.remove(out_file)
    except OSError:
        pass
    
    for ix, my_file in enumerate(input_files):
        my_dataset = pd.read_csv(my_file, sep=';')
        
        try:
            my_dataset['FORCE_2020_LITHOFACIES_LITHOLOGY'].to_hdf(
                out_file, key=my_file[:-4] + '_target') 
        except:
            my_target = pd.read_csv('leaderboard_test_target.csv', sep=';')
            my_target['FORCE_2020_LITHOFACIES_LITHOLOGY'].to_hdf(
                out_file, key=my_file[:-4] + '_target') 
                    
        if ix==0:    
            # Fitting the categorical encoders
            my_encoder = OrdinalEncoder()
            my_encoder.set_params(handle_unknown='use_encoded_value', 
                                  unknown_value=-1, 
                                  encoded_missing_value=-1).fit(
                                      my_dataset[['FORMATION']])

        my_dataset = (my_dataset.
                        pipe(replace_inf).
                        pipe(log_transform).
                        pipe(calculate_delta).
                        pipe(feature_selection))
        my_dataset.to_hdf(out_file, key=my_file[:-4])  
        
        my_dataset.to_hdf(out_file, key= my_file[:-4]) 

my_files = ['train.csv', 'leaderboard_test_features.csv', 'hidden_test.csv'] 

pre_processing_pipeline(input_files=my_files, out_file='ml_data.h5')