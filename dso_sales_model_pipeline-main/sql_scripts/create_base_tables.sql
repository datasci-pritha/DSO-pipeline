--Invoice Table
drop table if exists {{SCHEMA_NAME}}.invoice;
create table {{SCHEMA_NAME}}.invoice as
select 
    obligor_code
    , invoice_num
    , post_date::date
    , assignment_due_date::date as due_date
    , settlement_date::date
    , round(amount::numeric, 2) as amount
from {{SCHEMA_NAME}}.accounting_items ai 
where 
    doc_type = 'INVOICE'
    and char_length(obligor_code) = 7
    and substring(obligor_code, 1, 2) in ('40', '50', '60', '70')
order by 1, 2, 3;

-- Collection Table
drop table if exists {{SCHEMA_NAME}}.collection;
create table {{SCHEMA_NAME}}.collection as
select 
    obligor_code
    , client_doc_id as transaction_number
    , post_date::date
    , assignment_due_date::date as due_date
    , settlement_date::date
    , round(amount::numeric, 2) as amount
from {{SCHEMA_NAME}}.accounting_items ai 
where 
    doc_type = 'COLLECTION'
    and char_length(obligor_code) = 7
    and substring(obligor_code, 1, 2) in ('40', '50', '60', '70')
order by 1, 2, 3;

-- Adjustment Table
drop table if exists {{SCHEMA_NAME}}.adjustment;
create table {{SCHEMA_NAME}}.adjustment as
select 
    obligor_code
    , client_doc_id as transaction_number
    , post_date::date
    , assignment_due_date::date as due_date
    , settlement_date::date
    , round(amount::numeric, 2) as amount
from {{SCHEMA_NAME}}.accounting_items ai 
where 
    doc_type = 'MISC'
    and char_length(obligor_code) = 7
    and substring(obligor_code, 1, 2) in ('40', '50', '60', '70')
order by 1, 2, 3;