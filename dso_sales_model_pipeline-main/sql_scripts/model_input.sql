---------------------------------------------------------------------------------------------
--------------------------------------- MODEL METRICS ---------------------------------------
---------------------------------------------------------------------------------------------
drop table if exists {{SCHEMA_NAME}}.healthium_model_metrics;
create table {{SCHEMA_NAME}}.healthium_model_metrics(
    id serial,
    date date not null,
    model varchar(15) not null,
    sample varchar(15) not null,
    rmse decimal(15,2) not null,
    mae decimal(15,2) not null,
    r_sq decimal(4,2) not null,
    model_info json not null,
    rmse_actual decimal(15,2),
    mae_actual decimal(15,2),
    r_sq_actual decimal(4,2)
);

---------------------------------------------------------------------------------------------
---------------------------------- PAYMENT BEHAVIOUR Table ----------------------------------
---------------------------------------------------------------------------------------------

-- Captures Payment Behaviour
drop table if exists {{SCHEMA_NAME}}.payment_behaviour_table;
create table {{SCHEMA_NAME}}.payment_behaviour_table as (
select
	obligor_code,
	invoice_yyyymm,
	num_invoices_3m,
	-- sum_invoices_3m,
	paid_in_30_3m_avg,
	paid_in_60_3m_avg,
	paid_in_90_3m_avg,
	paid_in_60_3m_avg - paid_in_30_3m_avg as paid_in_31_60_3m_avg,
	paid_in_90_3m_avg - paid_in_60_3m_avg as paid_in_61_90_3m_avg,
	days_outs_3m_avg,
	num_invoices_6m,
	-- sum_invoices_6m,
	paid_in_30_6m_avg,
	paid_in_60_6m_avg,
	paid_in_90_6m_avg,
	paid_in_60_6m_avg - paid_in_30_6m_avg as paid_in_31_60_6m_avg,
	paid_in_90_6m_avg - paid_in_60_6m_avg as paid_in_61_90_6m_avg,
	paid_ever_6m - paid_in_90_3m_avg  as paid_gt_90_6m_avg,
	days_outs_6m_avg
from
(
select * from 
(
	select 
		obligor_code,
		invoice_yyyymm,
		first_snap,
		num_invoices_3m,
		-- sales_3m as sum_invoices_3m,
		case when (sales_3m+previous_balance_3) = 0  then 0 else paid_amt_3m/(sales_3m+previous_balance_3) end as paid_in_30_3m_avg,
		case when paid_in_60_3m_denom_1_2 + paid_in_60_3m_denom_2_3 = 0 then 0 else (paid_in_60_3m_num_1_2 + paid_in_60_3m_num_2_3)/(paid_in_60_3m_denom_1_2 + paid_in_60_3m_denom_2_3) end as paid_in_60_3m_avg,
		case when paid_in_90_3m_denom = 0 then 0 else (paid_amt_3m/paid_in_90_3m_denom) end as paid_in_90_3m_avg,
		case when sales_3m = 0 then null else (days_outs_3m_avg_num/sales_3m)*90 end as days_outs_3m_avg,
		num_invoices_6m,
		-- sales_6m as sum_invoices_6m,
		case when (sales_6m+previous_balance_6) = 0  then 0 else paid_amt_6m/(sales_6m+previous_balance_6) end as paid_in_30_6m_avg,
		case when (paid_in_60_6m_denom_1_2 + paid_in_60_6m_denom_2_3 + paid_in_60_6m_denom_3_4 + paid_in_60_6m_denom_4_5 + paid_in_60_6m_denom_5_6) = 0 	
			then 0 else (paid_in_60_6m_num_1_2 + paid_in_60_6m_num_2_3 + paid_in_60_6m_num_3_4 + paid_in_60_6m_num_4_5 + paid_in_60_6m_num_5_6)/(paid_in_60_6m_denom_1_2 + paid_in_60_6m_denom_2_3 + paid_in_60_6m_denom_3_4 + paid_in_60_6m_denom_4_5 + paid_in_60_6m_denom_5_6) 
			end as paid_in_60_6m_avg,
        case when (paid_in_90_6m_denom_1_2_3 + paid_in_90_6m_denom_2_3_4 + paid_in_90_6m_denom_3_4_5 + paid_in_90_6m_denom_4_5_6) = 0 	
			then 0 else (paid_in_90_6m_num_1_2_3 + paid_in_90_6m_num_2_3_4 + paid_in_90_6m_num_3_4_5 + paid_in_90_6m_num_4_5_6)/(paid_in_90_6m_denom_1_2_3 + paid_in_90_6m_denom_2_3_4 + paid_in_90_6m_denom_3_4_5 + paid_in_90_6m_denom_4_5_6) 
			end as paid_in_90_6m_avg,
        case when paid_ever_6m_denom = 0 then 0 else paid_amt_6m/paid_ever_6m_denom end as paid_ever_6m,
        case when sales_6m = 0 then null else (days_outs_6m_avg_num/sales_6m)*180 end as days_outs_6m_avg
	from(
		select
			obligor_code,
			invoice_yyyymm,
			coalesce((lag(previous_balance,2) over w),0) as previous_balance_3,
			coalesce((lag(previous_balance,5) over w),0) as previous_balance_6,
			case when invoice_yyyymm = first_value(invoice_yyyymm) over w then 1 else 0 end as first_snap,
			coalesce((lag(num_invoices,0) over w),0) + coalesce((lag(num_invoices,1) over w),0) + coalesce((lag(num_invoices,2) over w),0) as num_invoices_3m,
			coalesce((lag(sales,0) over w),0) + coalesce((lag(sales,1) over w),0) + coalesce((lag(sales,2) over w),0) as sales_3m,
			coalesce((lag(paid_amt,0) over w),0) + coalesce((lag(paid_amt,1) over w),0) + coalesce((lag(paid_amt,2) over w),0) as paid_amt_3m,
			coalesce((lag(previous_balance,0) over w),0) + coalesce((lag(previous_balance,1) over w),0) + coalesce((lag(previous_balance,2) over w),0) as previous_balance_3m,

			coalesce((lag(sales,0) over w),0) + coalesce((lag(sales,1) over w),0) + coalesce((lag(previous_balance,1) over w),0) as paid_in_60_3m_denom_1_2,
			coalesce((lag(sales,1) over w),0) + coalesce((lag(sales,2) over w),0) + coalesce((lag(previous_balance,2) over w),0) as paid_in_60_3m_denom_2_3,
			coalesce((lag(paid_amt,0) over w),0) + coalesce((lag(paid_amt,1) over w),0) as paid_in_60_3m_num_1_2,
			coalesce((lag(paid_amt,1) over w),0) + coalesce((lag(paid_amt,2) over w),0) as paid_in_60_3m_num_2_3,

			coalesce((lag(sales,0) over w),0) + coalesce((lag(sales,1) over w),0) + coalesce((lag(sales,2) over w),0) + coalesce((lag(previous_balance,2) over w),0) as paid_in_90_3m_denom,
			coalesce((lag(end_balance,0) over w),0) as days_outs_3m_avg_num,
		
			coalesce((lag(num_invoices,0) over w),0) + coalesce((lag(num_invoices,1) over w),0) + coalesce((lag(num_invoices,2) over w),0) + coalesce((lag(num_invoices,3) over w),0) + coalesce((lag(num_invoices,4) over w),0) + coalesce((lag(num_invoices,5) over w),0) as num_invoices_6m,
			coalesce((lag(sales,0) over w),0) + coalesce((lag(sales,1) over w),0) + coalesce((lag(sales,2) over w),0) + coalesce((lag(sales,3) over w),0) + coalesce((lag(sales,4) over w),0) + coalesce((lag(sales,5) over w),0) as sales_6m,
			coalesce((lag(paid_amt,0) over w),0) + coalesce((lag(paid_amt,1) over w),0) + coalesce((lag(paid_amt,2) over w),0) + coalesce((lag(paid_amt,3) over w),0) + coalesce((lag(paid_amt,4) over w),0) + coalesce((lag(paid_amt,5) over w),0) as paid_amt_6m,
			
			coalesce((lag(sales,0) over w),0) + coalesce((lag(sales,1) over w),0) + coalesce((lag(previous_balance,1) over w),0) as paid_in_60_6m_denom_1_2,
			coalesce((lag(sales,1) over w),0) + coalesce((lag(sales,2) over w),0) + coalesce((lag(previous_balance,2) over w),0) as paid_in_60_6m_denom_2_3,
			coalesce((lag(sales,2) over w),0) + coalesce((lag(sales,3) over w),0) + coalesce((lag(previous_balance,3) over w),0) as paid_in_60_6m_denom_3_4,
			coalesce((lag(sales,3) over w),0) + coalesce((lag(sales,4) over w),0) + coalesce((lag(previous_balance,4) over w),0) as paid_in_60_6m_denom_4_5,
			coalesce((lag(sales,4) over w),0) + coalesce((lag(sales,5) over w),0) + coalesce((lag(previous_balance,5) over w),0) as paid_in_60_6m_denom_5_6,
			coalesce((lag(paid_amt,0) over w),0) + coalesce((lag(paid_amt,1) over w),0) as paid_in_60_6m_num_1_2,
			coalesce((lag(paid_amt,1) over w),0) + coalesce((lag(paid_amt,2) over w),0) as paid_in_60_6m_num_2_3,
			coalesce((lag(paid_amt,2) over w),0) + coalesce((lag(paid_amt,3) over w),0) as paid_in_60_6m_num_3_4,
			coalesce((lag(paid_amt,3) over w),0) + coalesce((lag(paid_amt,4) over w),0) as paid_in_60_6m_num_4_5,
			coalesce((lag(paid_amt,4) over w),0) + coalesce((lag(paid_amt,5) over w),0) as paid_in_60_6m_num_5_6,
			
            coalesce((lag(sales,0) over w),0) + coalesce((lag(sales,1) over w),0) + coalesce((lag(sales,2) over w),0) + coalesce((lag(previous_balance,2) over w),0) as paid_in_90_6m_denom_1_2_3,
            coalesce((lag(sales,1) over w),0) + coalesce((lag(sales,2) over w),0) + coalesce((lag(sales,3) over w),0) + coalesce((lag(previous_balance,3) over w),0) as paid_in_90_6m_denom_2_3_4,
            coalesce((lag(sales,2) over w),0) + coalesce((lag(sales,3) over w),0) + coalesce((lag(sales,4) over w),0) + coalesce((lag(previous_balance,4) over w),0) as paid_in_90_6m_denom_3_4_5,
            coalesce((lag(sales,3) over w),0) + coalesce((lag(sales,4) over w),0) + coalesce((lag(sales,5) over w),0) + coalesce((lag(previous_balance,5) over w),0) as paid_in_90_6m_denom_4_5_6,
            coalesce((lag(paid_amt,0) over w),0) + coalesce((lag(paid_amt,1) over w),0) + coalesce((lag(paid_amt,2) over w),0) as paid_in_90_6m_num_1_2_3,
            coalesce((lag(paid_amt,1) over w),0) + coalesce((lag(paid_amt,2) over w),0) + coalesce((lag(paid_amt,3) over w),0) as paid_in_90_6m_num_2_3_4,
            coalesce((lag(paid_amt,2) over w),0) + coalesce((lag(paid_amt,3) over w),0) + coalesce((lag(paid_amt,4) over w),0) as paid_in_90_6m_num_3_4_5,
            coalesce((lag(paid_amt,3) over w),0) + coalesce((lag(paid_amt,4) over w),0) + coalesce((lag(paid_amt,5) over w),0) as paid_in_90_6m_num_4_5_6,
            
            coalesce((lag(sales,0) over w),0) + coalesce((lag(sales,1) over w),0) + coalesce((lag(sales,2) over w),0) + coalesce((lag(sales,3) over w),0) + coalesce((lag(sales,4) over w),0) + coalesce((lag(sales,5) over w),0) + coalesce((lag(previous_balance,5) over w),0) as paid_ever_6m_denom,
            
			coalesce((lag(end_balance,0) over w),0) as days_outs_6m_avg_num
		from {{SCHEMA_NAME}}.debtor_level_metrics
		WINDOW w AS (PARTITION BY obligor_code ORDER BY invoice_yyyymm)
		)t
	)q
where (obligor_code,invoice_yyyymm) in (select obligor_code,invoice_yyyymm from {{SCHEMA_NAME}}.debtor_level_metrics)
)l
);

-------------------------------------------------------------------------------------------------
------------------------------------- Sales Forecast Tables -------------------------------------
-------------------------------------------------------------------------------------------------

-- Captures Customer Invoicing Behaviour & Payment Behaviour
drop table if exists {{SCHEMA_NAME}}.sales_forecast_data_;
create table {{SCHEMA_NAME}}.sales_forecast_data_ as
select 
	obligor_code,
	invoice_yyyymm as invoice_yyyymm,
	coalesce(sales,0) as sales, 
	coalesce((lag(sales,1) over w),0) as sales_1,
	coalesce((lag(sales,2) over w),0) as sales_2,
	coalesce((lag(sales,3) over w),0) as sales_3,
	sum(coalesce(sales,0)) over (w rows between unbounded preceding and current row) as sales_till_now,
	coalesce((lag(sales,0) over w),0) + coalesce((lag(sales,1) over w),0) + coalesce((lag(sales,2) over w),0) as sales_3m,
	coalesce((lag(sales,0) over w),0) + coalesce((lag(sales,1) over w),0) as sales_2m,
	coalesce((lag(sales,0) over w),0) + coalesce((lag(sales,1) over w),0) + coalesce((lag(sales,2) over w),0) + coalesce((lag(sales,3) over w),0) + coalesce((lag(sales,4) over w),0) + coalesce((lag(sales,5) over w),0) as sales_6m,
	coalesce(lead(sales,3) over w,0) + coalesce(lead(sales,2) over w,0) + coalesce(lead(sales,1) over w,0) as next_3m_sales,
	coalesce(lead(sales,2) over w,0) + coalesce(lead(sales,1) over w,0) as next_2m_sales,
	coalesce(lead(sales,6) over w,0) + coalesce(lead(sales,5) over w,0) + coalesce(lead(sales,4) over w,0) + coalesce(lead(sales,3) over w,0) + coalesce(lead(sales,2) over w,0) + coalesce(lead(sales,1) over w,0) as next_6m_sales,
	coalesce(paid_amt,0) as collections, 
	sum(coalesce(paid_amt,0)) over (w rows between unbounded preceding and current row) as collections_till_now,
	coalesce((lag(paid_amt,0) over w),0) + coalesce((lag(paid_amt,1) over w),0) as collections_last_2m,
	coalesce((lag(paid_amt,0) over w),0) + coalesce((lag(paid_amt,1) over w),0) + coalesce((lag(paid_amt,2) over w),0) as collections_last_3m,
	coalesce((lag(paid_amt,0) over w),0) + coalesce((lag(paid_amt,1) over w),0) + coalesce((lag(paid_amt,2) over w),0) + coalesce((lag(paid_amt,3) over w),0) + coalesce((lag(paid_amt,4) over w),0) + coalesce((lag(paid_amt,5) over w),0) as collections_last_6m,
	coalesce(lead(end_balance,3)over w,0) as balance_3m,
	coalesce(lead(end_balance,2)over w,0) as balance_2m,
	coalesce(end_balance,0) as balance,
    coalesce((lag(end_balance,1) over w),0) as balance_1,
	coalesce((lag(end_balance,2) over w),0) as balance_2,
	coalesce((lag(end_balance,3) over w),0) as balance_3,
    sum(coalesce(num_invoices,0)) over (w rows between unbounded preceding and current row) as no_of_invoices_till_now
from {{SCHEMA_NAME}}.debtor_level_metrics
WINDOW w AS (PARTITION BY obligor_code ORDER BY invoice_yyyymm);

-- Adding extra information to sales forecast data
drop table if exists {{SCHEMA_NAME}}.sales_forecast_data;
create table {{SCHEMA_NAME}}.sales_forecast_data as 
select a.*, 
	case when a.sales = 0 then 0 else 1 end as sales_active_last_m,
	case when a.sales = 0 and a.balance = 0 and a.collections = 0 then 1 else 0 end as zero_sales_balance_collection,
	case when a.sales_6m = 0 and a.collections_last_6m = 0 then 1 else 0 end as no_sales_no_collection_last_6m,
	case when a.sales_2m = 0 then 0 else 1 end as sales_active_last_2m,
	case when a.sales_3m = 0 then 0 else 1 end as sales_active_last_3m,
	case when coalesce(b.end_balance, 0) = a.sales_3m then 1 else 0 end as paid_never_last_3m,
	case when coalesce(b.end_balance, 0) = a.sales_6m then 1 else 0 end as paid_never_last_6m
from
    {{SCHEMA_NAME}}.sales_forecast_data_ a 
join 
    {{SCHEMA_NAME}}.debtor_level_metrics b
on 
    a.obligor_code = b.obligor_code 
    and a.invoice_yyyymm = b.invoice_yyyymm 
WINDOW w AS (PARTITION BY  b.obligor_code ORDER BY b.invoice_yyyymm);

-- Gets the First Invoice Date of all Customers
drop table if exists {{SCHEMA_NAME}}.obligor_min_inv_date;
create table {{SCHEMA_NAME}}.obligor_min_inv_date as 
select 
	obligor_code,
	min(post_date) as min_inv_date
from {{SCHEMA_NAME}}.accounting_items
where
    -- substring(invoice_num,1,2) != '44'
    -- and 
	doc_date > '2013-01-01'
    and (rtrim(substring(private::varchar ,56,2),'\"}"') != 'UL' or private ->> 'client_doc_type' != 'UL')
    and char_length(obligor_code) = 7
    and substring(obligor_code, 1, 2) in ('40', '50', '60', '70')
group by 1;

-- To make data consistent
alter table {{SCHEMA_NAME}}.sales_forecast_data add column obligor_min_inv_date date;
update {{SCHEMA_NAME}}.sales_forecast_data a set obligor_min_inv_date = case when(b.min_inv_date < '2017-04-01') then '2017-04-01' else b.min_inv_date END
from {{SCHEMA_NAME}}.obligor_min_inv_date b 
where a.obligor_code = b.obligor_code;

-- Function to Convert Snaps to their Last Day of the Snap
CREATE OR REPLACE FUNCTION last_day_(DATE)
RETURNS DATE AS
$$
  SELECT (date_trunc('MONTH', $1) + INTERVAL '1 MONTH - 1 day')::DATE;
$$ LANGUAGE 'sql' IMMUTABLE STRICT;

-- Gets the Obligor Age as of that particular Snap
alter table {{SCHEMA_NAME}}.sales_forecast_data add column obligor_age int;
update {{SCHEMA_NAME}}.sales_forecast_data set obligor_age = last_day_(to_date(invoice_yyyymm,'YYYYMMDD')) - obligor_min_inv_date ;

-- Adding Max Snap to sales_forecast_data
alter table {{SCHEMA_NAME}}.sales_forecast_data add column max_snap text;
update {{SCHEMA_NAME}}.sales_forecast_data a set max_snap = b.max_yyyymm
from 
( select 
	obligor_code,
	invoice_yyyymmmax as max_yyyymm
from {{SCHEMA_NAME}}.obligors_min_max_snaps
)b
where a.obligor_code = b.obligor_code;

-- Adding Beyond Max Snap to sales_forecast_data
alter table {{SCHEMA_NAME}}.sales_forecast_data add column beyond_max_snap int;
update {{SCHEMA_NAME}}.sales_forecast_data a set beyond_max_snap = case when a.invoice_yyyymm > a.max_snap then 1 else 0 end;

-- Dropping Unnecessary Fields
alter table {{SCHEMA_NAME}}.sales_forecast_data 
	drop obligor_min_inv_date,
	drop max_snap;
