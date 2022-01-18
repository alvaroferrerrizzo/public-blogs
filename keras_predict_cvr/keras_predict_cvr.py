from google.cloud import bigquery
client = bigquery.Client(project=project_id)

df_main = client.query('''
select
    features
from core_table

''' ).to_dataframe()


#mostly NaNs
df_all=df_all.drop(columns={'avg_battery_level_kick_scooter','avg_battery_level_bici',
  'avg_battery_level_moto'})

#more than one ride
df_riders = df_main.loc[df_main['total_rides_before_mv']>=1]

#mostly NaNs
categorical_cols = ['segment', 'area']
#categorical_cols = ['segment', 'week_day']

#change met_ride, demand_met, plan_activated
binary_cols = ['met_ride', 'demand_met', 'plan_activated']

df_all['met_ride'] = df_all['met_ride'].apply(lambda x: 1 if x == 'yes' else 0)
df_all['demand_met'] = df_all['demand_met'].apply(lambda x: 1 if x == 'met' else 0)
df_all['plan_activated'] = df_all['plan_activated'].apply(lambda x: 1 if x == 'True' else 0)

df_categorical = pd.get_dummies(df_all, columns = categorical_cols, drop_first = True)

#dictionary assigning h3_ids with the operating area
df_sa_h3 = df_main.groupby(['h3','area']).size().reset_index().rename(columns={0:'count'})
h3_area_dict = {}
for idx,rw in df_sa_h3.iterrows():
    h3_area_dict[rw.h3] = rw.area

#impute and scale numerical values
df_categorical["avg_ride_distance"].fillna(
  (df_categorical["avg_ride_distance"].mean()), inplace=True)

scaler = StandardScaler()
df_categorical[['credit', 'promotion','avg_ride_distance']] = scaler.fit_transform(df_categorical[['credit', 'promotion','avg_ride_distance']])

#met_ride is our target variable
X = df_categorical.drop(columns={'met_ride'})
y = df_categorical['met_ride']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train.shape)
print(X_test.shape)

(408061, 40)
(200986, 40)

for categorical_var in X_train.select_dtypes(include=['object']):
    
    cat_emb_name= categorical_var.replace(" ", "")+'_Embedding'
  
    no_of_unique_cat  = X_train[categorical_var].nunique()
    embedding_size = int(min(np.ceil((no_of_unique_cat)/2), 50 ))
  
    print('Categorica Variable:', categorical_var,
        'Unique Categories:', no_of_unique_cat,
        'Embedding Size:', embedding_size)

input_models=[]
output_embeddings=[]
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

for categorical_var in X_train.select_dtypes(include=['object']):
    
    #Name of the categorical variable that will be used in the Keras Embedding layer
    cat_emb_name= categorical_var.replace(" ", "")+'_Embedding'
    print(cat_emb_name)
    # Define the embedding_size
    no_of_unique_cat  = X_train[categorical_var].nunique()
    embedding_size = int(min(np.ceil((no_of_unique_cat)/2), 50 ))
  
    #One Embedding Layer for each categorical variable
    input_model = Input(shape=(1,))
    output_model = Embedding(no_of_unique_cat, embedding_size, name=cat_emb_name)(input_model)
    output_model = Reshape(target_shape=(embedding_size,))(output_model)    
  
    #Appending all the categorical inputs
    input_models.append(input_model)
  
    #Appending all the embeddings
    output_embeddings.append(output_model)

#Other non-categorical data columns (numerical). 
#We define single another network for the other columns and add them to our models list.
input_numeric = Input(shape=(len(X_train.select_dtypes(include=['number']).columns.tolist()),))

print(len(X_train.select_dtypes(include=['number']).columns.tolist()))

input_models.append(input_numeric)
output_embeddings.append(input_numeric)

#At the end we concatenate altogther and add other Dense layers
output = Concatenate()(output_embeddings)
output = Dense(1000, kernel_initializer="uniform")(output)
output = Activation('relu')(output)
output= Dropout(0.4)(output)
output = Dense(300, kernel_initializer="uniform")(output)
output = Activation('relu')(output)
output= Dropout(0.3)(output)
output = Dense(1, activation='sigmoid')(output)

model = Model(inputs=input_models, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(
    input_list_train,y_train,validation_data=(input_list_val,y_test) ,
    epochs =  20 , batch_size = 1000)

emb_layer = model.get_layer('h3_Embedding')
(w,) = emb_layer.get_weights()