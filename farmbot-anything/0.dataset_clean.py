import pandas as pd
import os



df = pd.read_csv("/data/jyp/farmbot.ai/2.shapenet/shapenetsem/metadata.csv")
# for unique in df['category'].unique():
#     print(unique)
    
df = df[df['category'].notna()]
df = df[df['up'].notna()]
df = df[df['front'].notna()]
# Function to get the first word before comma
def get_first_word(text):
    if isinstance(text, str):
        return text.split(",")[0]
    else:
        return text
    
df['category'] = df['category'].apply(get_first_word)
# Replace specific strings in 'category' with first word from 'wnlemmas'
strings_to_replace = ['_StanfordSceneDBModels', '_PilotStudyModels', '_GeoAutotagEvalSet', '_OIMwhitelist', '_RandomSetStudyModels', '_Attributes', '_BAD', '_EvalSetInScenes', '_EvalSetNoScenesNoPrior']

for string in strings_to_replace:
    df['category'] = df.apply(lambda row: get_first_word(row['wnlemmas']) if isinstance(row['category'], str) and string in row['category'] else row['category'], axis=1)
df['category'] = df['category'].str.title().str.replace(' ', '')
df = df.loc[df['name'] != 'nan']

# Function to round the float values in a vector
def round_vector(vector):
    # Split the string into components
    components = vector.split('\\,')
    # Convert the components to floats, round them, and convert them back to strings
    components = [str(round(float(component))) for component in components]
    # Join the components back together
    return '\\,'.join(components)

# Apply the function to the 'up' column
df['up'] = df['up'].apply(round_vector)
df['front'] = df['front'].apply(round_vector)
df = df[df['unit'].notna()]
df2 = df[df['weight'].notna()]
df3 = df2[df2['surfaceVolume'].notna()]
df3 = df3[df3['category'].notna()]
df4 = df3.drop(columns=['wnsynset', 'wnlemmas', 'name', 'tags'])
df4 = df4.reset_index()
df4 = df4.rename(columns={'index': 'old_index'})
df4['category_id'] = pd.factorize(df4['category'])[0]
df4.to_csv("/data/jyp/farmbot.ai/2.shapenet/shapenet_rendered/metadata.csv")