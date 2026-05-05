-- Restarting the identity for the recommended_credit_limit_histories table
truncate public.recommended_credit_limit_histories restart identity;

-- Inserting Data into recommended_credit_limit_histories table from ratings_copy
insert
	into
	public.recommended_credit_limit_histories (client_id ,
	sys_obligor_code ,
	recommended_limit ,
	date)
select
	6,
	concat('6_', obligor_code),
	recommended_credit_limit,
	date
from
	public.ratings_copy rc;

-- Updating the ratings table using Obligors table & ratings table 
insert
	into
	public.ratings(client_id ,
	sys_obligor_code ,
	rating_5_scale ,
	rating ,
	date)
select
	distinct r.client_id ,
	o.sys_obligor_code ,
	rating_5_scale ,
	rating ,
	"date"
from
	public.ratings r
inner join 
	public.obligors o 
on
	r.sys_obligor_code = o.parent_sys_obligor_code
	and association_dim_id is not null;

-- Updating the date in the ratings table
update public.ratings set date = date - interval '1 month';
