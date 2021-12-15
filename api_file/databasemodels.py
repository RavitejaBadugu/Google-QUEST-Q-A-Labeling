from psycopg2 import sql
import math
from utils.config_reader import database_settings
from utils.y_labels import y_columns
from utils.connection import cursor,connector

FEATURE_TABLE=database_settings.TABLES_FEATURES

PREDICTION_TABLE=database_settings.TABLES_PREDICTIONS

CREATE_FEATURES=y_columns

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
    total=0
    if total>=n:
        page_no=math.floor(n/5)
        q=sql.SQL("select * from {name} OFFSET {offrows} limit {show_rows}").format(
            name=sql.Identifier(FEATURE_TABLE),
            offrows=sql.Placeholder(),
            show_rows=sql.Placeholder()
        )
        cursor.execute(q,(5*page_no,(5*page_no)+5))
    else:
        cursor.execute('select * from features')
    return cursor.fetchall()