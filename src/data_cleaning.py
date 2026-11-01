import pandas as pd

def clean_data(file_path="data/Reviews.csv"):
    df = pd.read_csv(file_path)
    
    # Drop unnecessary columns
    df = df.drop(['Id', 'UserId', 'ProfileName'], axis=1)
    
    # Drop missing values
    df.dropna(subset=['Summary'], inplace=True)
    
    # Filter neutral ratings
    df = df[df['Score'] != 3]
    
    # Create target and text features
    df.loc[:, 'Sentiment'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)
    df.loc[:, 'combined_text'] = df['Summary'] + " " + df['Text']
    
    return df

