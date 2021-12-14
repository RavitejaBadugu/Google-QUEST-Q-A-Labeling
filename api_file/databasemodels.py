import psycopg2
from psycopg2 import sql
import math
from config_reader import database_paras
import time
from datetime import datetime
from psycopg2.extras import RealDictCursor

connector=None
while True:
    try:
        connector=psycopg2.connect(dbname=database_paras["DATABASE"],
                                    user=database_paras['USERNAME'],
                                    password=database_paras['PASSWORD'],
                                    host=database_paras['HOSTNAME'],
                                    port=database_paras['PORT'],
                                    cursor_factory=RealDictCursor)

        cursor=connector.cursor()
        print("WE MADE A SUCCESSFUL CONNECTION TO THE DATABASE")
        break
    except Exception as error:
        print("CONNECTION FAILED")
        print("ERROR : ",error)
        time.sleep(10)

FEATURE_TABLE=database_paras['TABLES']['FEATURES']

PREDICTION_TABLE=database_paras['TABLES']['PREDICTIONS']

CREATE_FEATURES=[
        'question_asker_intent_understanding',
       'question_body_critical', 'question_conversational',
       'question_expect_short_answer', 'question_fact_seeking',
       'question_has_commonly_accepted_answer',
       'question_interestingness_others', 'question_interestingness_self',
       'question_multi_intent', 'question_not_really_a_question',
       'question_opinion_seeking', 'question_type_choice',
       'question_type_compare', 'question_type_consequence',
       'question_type_definition', 'question_type_entity',
       'question_type_instructions', 'question_type_procedure',
       'question_type_reason_explanation', 'question_type_spelling',
       'question_well_written', 'answer_helpful',
       'answer_level_of_information', 'answer_plausible', 'answer_relevance',
       'answer_satisfaction', 'answer_type_instructions',
       'answer_type_procedure', 'answer_type_reason_explanation',
       'answer_well_written']

def CREATE_TABLES():
    query=sql.SQL("""
          DROP TABLE IF EXISTS {table};
          CREATE TABLE if not exists {table} (
              ID SERIAL PRIMARY KEY,
              QUESTIONS text NOT NULL,
              ANSWERS text NOT NULL,
              TITLE text NOT NULL,
              POSTED_DATE TIMESTAMP Not NULL
          )
          """
    ).format(table=sql.Identifier(FEATURE_TABLE))

    print(query.as_string(connector))
    cursor.execute(query)
    # create table to keep the predictions
    query=sql.SQL("""
                  DROP TABLE IF EXISTS {table};
                  CREATE TABLE if not exists {table} 
                    ( ID SERIAL PRIMARY KEY,
                      PREDICTION_DATE TIMESTAMP Not NULL,
                    {field}
                    )
                """
                ).format(table=sql.Identifier(PREDICTION_TABLE),
                field=sql.SQL(', ').join(
                    [sql.SQL(' ').join([sql.Identifier(col), sql.SQL('decimal'), sql.SQL('Not Null')])
                        for col in CREATE_FEATURES]
                    )
                )

    print(query.as_string(connector))
    cursor.execute(query)
    connector.commit()
    print(f"BOTH TABLES {FEATURE_TABLE} AND {PREDICTION_TABLE} ARE CREATED")

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



if __name__ == '__main__':
    CREATE_TABLES()
    add_features('jsrhfbvz','eFDSHVcjhWSDfv','jhwefvbcwkf',datetime.now())
    print(get_n_features(5))
    if connector is not None:
        cursor.close()
        connector.close()