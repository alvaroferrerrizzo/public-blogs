import pandas as pd
from google.cloud import bigquery
import pyarrow

client = bigquery.Client(project = 'project-id')

def update_tablee(event, context):
    sql = """
    CREATE OR REPLACE TABLE `project-id.staging.table_name` AS
    SELECT * FROM EXTERNAL_QUERY("project-id.eu.prod-ddbb", 
    '''
    SELECT
    user_id,
    last_balance,
    last_transaction_id
    from user
    where date(updated_at) >= now() - interval '1 day'

    ''' )
    ;

    MERGE `project-id.prod.table_name` T
    USING `project-id.staging.table_name` S
    ON T.user_id = S.user_id

    WHEN NOT MATCHED THEN
    INSERT(user_id,last_balance,last_transaction_id)
    VALUES(user_id,last_balance,last_transaction_id)

    WHEN MATCHED THEN
    UPDATE SET 
        T.user_id = S.user_id,
        T.last_balanc = S.last_balance,
        T.last_transaction_id = S.last_transaction_id
        """
    job = client.query(sql)
    result = job.result()

    return 'rows updated',200