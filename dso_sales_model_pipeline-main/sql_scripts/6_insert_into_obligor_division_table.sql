-- Insert into obligor division credit limit Table
Insert into public.obligor_division_credit_limit
with obl_sales as
(select 	
	right(parent_sys_obligor_code, -2) as obligor_code
	, to_char(post_date, 'YYYYMM') as yyyymm
	, division
	, sum(amount) as sales
from public.sales_ledger usl  
where 
	char_length(parent_sys_obligor_code) = 9
	and substring(parent_sys_obligor_code, 3, 2) in ('40', '50', '60', '70')
	and post_date >= current_date - interval '10 months'
group by 1, 2, 3
order by 2, 3
)
, division as
(select 
	distinct division
from obl_sales 
where 
	division notnull
)
, snaps as
(select 
	distinct yyyymm
from obl_sales
)
, obligor as
(select
	distinct right(parent_sys_obligor_code, -2) as obligor_code
from public.sales_ledger
where 
	char_length(parent_sys_obligor_code) = 9
	and substring(parent_sys_obligor_code, 3, 2) in ('40', '50', '60', '70')
union 
select
	distinct obligor_code
from public.accounting_items ai 
where 
	char_length(obligor_code) = 7
	and substring(obligor_code, 1, 2) in ('40', '50', '60', '70')
)
, obl_division_snaps as
(select *
from obligor, division, snaps
order by obligor_code, division, yyyymm
)
, obl_division_snaps_sales as
(select
	obligor_code 
	, division
	, yyyymm
	, case when sales isnull then 0 else sales end as monthly_division_sales 
from 
	obl_division_snaps
left join
	obl_sales using(obligor_code, division, yyyymm)
)
, all_division_snaps as
(select *
	, sum(monthly_division_sales) over (PARTITION BY obligor_code, division ORDER BY yyyymm ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) as division_sales_last_6m
from obl_division_snaps_sales
)
, customer_snaps as
(select
	obligor_code 
	, yyyymm
	, sum(monthly_division_sales) as monthly_sales
from obl_division_snaps_sales
group by 1, 2
order by 1, 2
)
, all_customer_snaps as
(select *
	, sum(monthly_sales) over (PARTITION BY obligor_code ORDER BY yyyymm ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) as obligor_sales_last_6m 
	, sum(monthly_sales) over (PARTITION BY obligor_code ORDER BY yyyymm) as cum_sum
from customer_snaps
)
, all_snaps as
(select
	obligor_code 
	, division
	, (to_date(yyyymm, 'YYYYMM') + interval '1 month')::date as date
	, monthly_division_sales
	, division_sales_last_6m
	, obligor_sales_last_6m
	, case when obligor_sales_last_6m != 0 then division_sales_last_6m / obligor_sales_last_6m 
		else null 
	end as percent_sales
from 
	all_division_snaps
left join
	all_customer_snaps using (obligor_code, yyyymm)
where 
	yyyymm = to_char(current_date - interval '1 month', 'YYYYMM')
order by 1, 3, 2
)
, obligor_name as
(select
	obligor_code
	, max(obligor_name) as obligor_name
from public.obligors 
where 
	char_length(parent_sys_obligor_code) = 9
	and substring(parent_sys_obligor_code, 3, 2) in ('40', '50', '60', '70')
group by 1
order by 1
)
select 
	a.obligor_code 
	, obligor_name
	, division
	, to_char(a.date, 'YYYYMMDD') as recommended_date
	, to_char((a.date + interval '4 day')::date, 'YYYYMMDD') as valid_from
	, to_char((a.date + interval '3 day' + interval '1 month')::date, 'YYYYMMDD') as valid_to
	, division_sales_last_6m as division_sales_last_6m
	, obligor_sales_last_6m
	, percent_sales as division_percentage_sales
	, recommended_credit_limit as obligor_recommended_credit_limit
	, round(case when (division_sales_last_6m <= 0 or division = '00' or obligor_sales_last_6m <= 0) then 1 
		else (case when (recommended_credit_limit isnull or u.sales_multiplier = 0 or recommended_credit_limit <= 1) then 1 
				else (case when (obligor_sales_last_6m isnull or obligor_sales_last_6m = 0) then recommended_credit_limit/9
						else (case when (percent_sales isnull or percent_sales = 0)  then 1 
								else percent_sales * recommended_credit_limit
							end)
					end)
			end)
		end) as division_recommended_credit_limit
	, case when division = '00' then 'Common Division'
		else (case when obligor_sales_last_6m <= 0 then 'Last 6 Months sales is less than zero'
				else (case when division_sales_last_6m <= 0 then 'Division sales is less than zero'
						else (case when recommended_credit_limit isnull then 'Not an active Customer'
								else (case when (u.sales_multiplier = 0) then 'Sales Multiplier is Zero'
										else (case when recommended_credit_limit <= 1 then 'Recommended Credit Limit is less than 5000'
												else (case when percent_sales = 0 then 'No Sales in the last 6 Months for that division'
														else (case when ((obligor_sales_last_6m isnull or obligor_sales_last_6m = 0) and recommended_credit_limit > 0) then 'No Sales in the last 6 months but has collection in last 6 months'
																else null 
															end)
													end)
											end)
									end)
							end)
					end)
			end)
	end as floored_credit_limit_comments
from 
	all_snaps a
join 
	obligor_name o using(obligor_code)
left join
	public.model_outputs_copy u
on 
	a.obligor_code = u.obligor_code 
	and a.date = u.date
order by 1, 2, 3, 4;
