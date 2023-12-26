# Author: Junyong Park
import pandas as pd
import os
import yaml

# Function to get the first word before comma
def get_first_word(text):
        if isinstance(text, str):
            return text.split(",")[0]
        else:
            return text

# Main function
def main(cfg):
    '''
    This function reads the csv file and cleans the data.
    args:
        cfg: config file

    returns:
        None
    '''
    # Read the csv file
    df = pd.read_csv(cfg['0_input'])
    
    # Drop the columns that are not available
    df = df[df['category'].notna()]
    df['category'] = df['category'].apply(get_first_word)
    
    # Drop up and front columns that are not available
    df = df[df['up'].notna()]
    df = df[df['front'].notna()]
        
    # Replace specific strings in 'category' with first word from 'wnlemmas'
    strings_to_replace = [
        '_StanfordSceneDBModels', 
        '_PilotStudyModels', 
        '_GeoAutotagEvalSet', 
        '_OIMwhitelist', 
        '_RandomSetStudyModels', 
        '_Attributes', 
        '_BAD', 
        '_EvalSetInScenes', 
        '_EvalSetNoScenesNoPrior']

    # Replace specific strings in 'category' with first word from 'wnlemmas'
    for string in strings_to_replace:
        df['category'] = df.apply(
            lambda row: get_first_word(row['wnlemmas']) \
                if isinstance(row['category'], str) \
                    and string in row['category'] else row['category'], axis=1)
    df['category'] = df['category'].str.title().str.replace(' ', '')
    df = df.loc[df['name'] != 'nan']

    # Function to round the vector
    def round_vector(vector):
        components = vector.split('\\,')
        components = [str(round(float(component))) for component in components]
        return '\\,'.join(components)

    # Apply the function to the 'up' column
    df['up'] = df['up'].apply(round_vector)
    df['front'] = df['front'].apply(round_vector)

    # Drop the columns that are not available
    df = df[df['unit'].notna()]
    df = df[df['weight'].notna()]
    df = df[df['surfaceVolume'].notna()]
    df = df[df['category'].notna()]
    df = df.drop(columns=['wnsynset', 'wnlemmas', 'name', 'tags'])
    
    # Reset the index and rename the column
    df = df.reset_index()
    df = df.rename(columns={'index': 'old_index'})
    
    # Add a new column with the new index
    df['category_id'] = pd.factorize(df['category'])[0]

    # Save the dataframe to a csv file
    df.to_csv(cfg['0_output'])

if __name__ == '__main__':
    with open("./config/config.yml", 'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg)