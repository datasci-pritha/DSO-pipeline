-- Deleting the model_outputs_copy table
delete from public.model_outputs_copy;

-- Restarting the identity of the model_outputs table
truncate public.model_outputs restart identity;