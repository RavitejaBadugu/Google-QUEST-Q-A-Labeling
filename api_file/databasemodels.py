import psycopg2
from psycopg2 import sql
from config_reader import database_paras
from fastapi import HTTPException,status

connector=None
try:
    connector=psycopg2.connect(dbname=database_paras["DATABASE"],
                                user=database_paras['USERNAME'],
                                password=database_paras['PASSWORD'],
                                host=database_paras['HOSTNAME'],
                                port=database_paras['PORT'])

    cursor=connector.cursor()
except:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED,detail="data base is not started")

        
query=sql.SQL("""
      CREATE TABLE if not exists {table} (
          ID SERIAL PRIMARY KEY,
          QUESTIONS text NOT NULL,
          ANSWERS text NOT NULL,
          TITLE text NOT NULL
      )
      """
).format(table=sql.Identifier(database_paras['TABLES']['FEATURES']))


print(query.as_string(connector))
cursor.execute(query)

create_features=[
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

query=sql.SQL("""
              CREATE TABLE if not exists {table} 
                ( ID SERIAL PRIMARY KEY,
                {field}
                )
            """
            ).format(table=sql.Identifier(database_paras['TABLES']['PREDICTIONS']),
            field=sql.SQL(', ').join(
                [sql.SQL(' ').join([sql.Identifier(col), sql.SQL('decimal'), sql.SQL('Not Null')])
                    for col in create_features]
                )
            )
            
print(query.as_string(connector))
cursor.execute(query)

connector.commit()


if connector is not None:
    cursor.close()
    connector.close()