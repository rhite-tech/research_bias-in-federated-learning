import pandas as pd
import os

current_path = os.getcwd()

def get_state_metadata():
    """
    Obtain metadata of dataset per state, i.e. dataset size, number 
    of male females, whites, non-whites, and how many of them have an 
    income below or above 50K

    input: list with dataframe for all states
    output: one dataframe with metadata per frame
    """

    state_metadata = pd.DataFrame(columns=['State', 'Size', '#Males', '#Females', '#Whites', '#Non_whites', 'Mean Age',  '#>50k', '#<50k', '%>50k', 'female_>50k', 'male_>50k', 'white_>50k', 'non_white_>50k'])
    data_folderpath = os.path.join(current_path, 'acs_data')
    file_list = os.listdir(data_folderpath)
    list_state_df = [pd.read_csv(os.path.join(data_folderpath, file)) for file in file_list]
    
    for df in list_state_df:
        state = df.loc[0, 'State']
        size = len(df)
        males = df['Sex'].value_counts()[1]
        females = df['Sex'].value_counts()[2]
        whites = df['Race'].value_counts()[1]
        non_whites = size - whites

        assert size == males + females, "Sizes don't add up"

        # calculate percentage that earns above/below 50k
        below_50k = df.groupby('Income').size()['<=50K']
        above_50k = df.groupby('Income').size()['>50K']
        percentage = (above_50k / (above_50k + below_50k)) * 100

        # count females and males with income above/below 50k
        count_sex = df[['Sex', 'Income']].value_counts(ascending=True).reset_index(name='count')
        female_above_50k = count_sex.loc[(count_sex['Sex'] == 2 ) & (count_sex['Income'] == '>50K')]['count'].sum()
        male_above_50k = count_sex.loc[(count_sex['Sex'] == 1 ) & (count_sex['Income'] == '>50K')]['count'].sum()
        
        # count non-whites and whites with income above/below 50k
        count_race = df[['Race', 'Income']].value_counts(ascending=True).reset_index(name='count')
        non_white_above_50k = count_race.loc[(count_race['Race'].isin([2, 3, 4, 5, 6, 7, 8, 9])) & (count_race['Income'] == '>50K')]['count'].sum()
        white_above_50k = count_race.loc[(count_race['Race'] == 1) & (count_race['Income'] == '>50K')]['count'].sum()
        
        # calculate mean age per state
        mean_age = df.loc[:, 'Age'].mean().round(2)

        # create dataframe with all relevant variables
        new_row = {'State': state, 'Size': size, '#Males':males, '#Females':females, '#Whites':whites, '#Non_whites':non_whites, 'Mean Age':mean_age, '#>50k': above_50k, '#<50k': below_50k, 'female_>50k':female_above_50k, 'male_>50k':male_above_50k, 'white_>50k': white_above_50k, 'non_white_>50k': non_white_above_50k,'%>50k':percentage}
        state_metadata.loc[len(state_metadata)] = new_row
    return state_metadata


def check_all_classes(train_data, val_data, test_data):
    """
    Check whether all classes of 'Sex', 'Race' and 'Income' occur in the dataset

    input: train, val and test datasets
    output: For every feature, what classes are missing within the train, val and test set
    """

    def subset_to_dataframe(subset):
        features_list = []
        labels_list = []
        for idx in range(len(subset.indices)):
            features, label = subset.dataset[idx]
            features_list.append(features.tolist())  # features is already an array
            labels_list.append(label.item())  # label is already a scalar
        df = pd.DataFrame(features_list, columns=['Sex', 'Race', 'Class of Worker', 'Education', 'Age', 'Marital Status', 'Occupation', 'Place of Birth', 'Worked hours'])
        df['Income'] = labels_list
        return df
            
    df_train = subset_to_dataframe(train_data)
    df_val = subset_to_dataframe(val_data)
    df_test = subset_to_dataframe(test_data)

    features_to_check = ['Sex', 'Race', 'Income']
    expected_classes = {
        'Sex': [1, 2],
        'Race': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'Income': [0, 1]
    }

    train_missing = {}
    val_missing = {}
    test_missing = {}

    for feature in features_to_check:
        classes_train = df_train[feature].value_counts().index.tolist()
        classes_val = df_val[feature].value_counts().index.tolist()
        classes_test = df_test[feature].value_counts().index.tolist()

        missing_classes_train = set(expected_classes[feature]) - set(classes_train)
        missing_classes_val = set(expected_classes[feature]) - set(classes_val)
        missing_classes_test = set(expected_classes[feature]) - set(classes_test)

        if missing_classes_train: 
            train_missing[feature] = missing_classes_train
        if missing_classes_val:
            val_missing[feature] = missing_classes_val
        if missing_classes_test:
            test_missing[feature] = missing_classes_test

    return train_missing, val_missing, test_missing