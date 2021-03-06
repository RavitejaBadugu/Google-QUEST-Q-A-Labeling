import psycopg2
from psycopg2 import sql
from psycopg2.extras import DictCursor
import math
from api_file.utils.config_reader import database_settings
from api_file.utils.y_labels import y_columns

FEATURE_TABLE=database_settings.TABLES_FEATURES

PREDICTION_TABLE=database_settings.TABLES_PREDICTIONS

CREATE_FEATURES=y_columns

connector=None
connector=psycopg2.connect(dbname=database_settings.DATABASE,
                            user=database_settings.DATABASE_USER,
                            password=database_settings.DATABASE_PASSWORD,
                            host=database_settings.POSTGRES_DATABASE_HOSTNAME,
                            port=database_settings.POSTGRES_PORT,
                            cursor_factory=DictCursor)

cursor=connector.cursor()

#add values
def add_features(question,answer,title,time_posted):
    query=sql.SQL("""
                insert into {name} (QUESTIONS,ANSWERS,TITLE,POSTED_DATE)
                VALUES ({question},{answer},{title},{time_posted})

            """).format(name=sql.Identifier(FEATURE_TABLE),
                        question=sql.Placeholder(),
                        answer=sql.Placeholder(),
                        title=sql.Placeholder(),
                        time_posted=sql.Placeholder())
    cursor.execute(query,(question,answer,title,time_posted))
    connector.commit()

# add predictions
def add_predictions(model_predictions,prediction_time):
    query=sql.SQL("""
                insert into {name} ({fields},PREDICTION_DATE)
                VALUES ({values})

            """).format(name=sql.Identifier(PREDICTION_TABLE),
                        fields=sql.SQL(', ').join([sql.Identifier(col)
                        for col in CREATE_FEATURES]),
                        values=sql.Placeholder()*(len(CREATE_FEATURES)+1)
                        )
    assert type(model_predictions)==list
    cursor.execute(query,tuple(model_predictions+[prediction_time]))
    connector.commit()

def get_n_features(n):
    try:
        q=sql.SQL("select count(*) from {table}").format(table=sql.Identifier(FEATURE_TABLE))
        cursor.execute(q)
        total=cursor.fetchall()[0]['count']
        #print(f"total is {total}")
        page_no=math.floor(n/5)
        q=sql.SQL("select * from {name} OFFSET {offrows} limit {show_rows}").format(
                name=sql.Identifier(FEATURE_TABLE),
                offrows=sql.Placeholder(),
                show_rows=sql.Placeholder()
            )
        if total>=n:
            cursor.execute(q,(5*page_no,(5*page_no)+5))
        else:
            if total>=5:
                cursor.execute(q,(total-5,total+5))
            else:
                q=sql.SQL("select * from {table}").format(table=sql.Identifier(FEATURE_TABLE))
                cursor.execute(q)
        return cursor.fetchall()
    except:
        return 'no data is uploaded before'