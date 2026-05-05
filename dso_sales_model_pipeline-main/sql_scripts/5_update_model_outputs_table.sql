-- Inserting data into model_outputs table using model_outputs_copy 
insert
	into
	public.model_outputs (client_id ,
	sys_obligor_code ,
	parent_sys_obligor_code ,
	predicted_dso ,
	predicted_sales,
	sales_multiplier,
	rating,
	recommended_limit,
	date)
select
	6,
	concat('6_', obligor_code),
	concat('6_', obligor_code),
	dso_pred,
	sales_pred,
	sales_multiplier,
	rating,
	recommended_credit_limit,
	date
from
	public.model_outputs_copy;