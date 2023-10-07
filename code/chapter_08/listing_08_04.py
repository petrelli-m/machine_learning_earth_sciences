def calculate_delta(dataFrame):
    delta_features = ['CALI', 'log_RMED', 'log_RDEP', 'RHOB', 'DTC', 'DRHO', 'log_GR' , 'NPHI', 'log_PEF', 'SP']
    wells = dataFrame['WELL'].unique()
    for my_feature in delta_features:
        values = []
        for well in wells:
            col_values = dataFrame[dataFrame['WELL'] == well][my_feature].values
            col_values_ = np.array([col_values[0]]+list(col_values[:-1]))
            delta_col_values = col_values-col_values_
            values = values + list(delta_col_values)
        dataFrame['Delta_' + my_feature] = values
    return dataFrame