-- Insert data into ratings table from ratings_copy table
insert
	into
	public.ratings(client_id,
	sys_obligor_code ,
	rating_5_scale,
	rating,
	date)
select
	6,
	concat('6_', obligor_code),
	rating_5_scale ,
	rating ,
	date
from
	public.ratings_copy rc;

-- Restarting the Identity for recommended_credit_limits table
truncate public.recommended_credit_limits restart identity;
