# Configuration
Configurations are stored in a JSON file [`config.json`](../libera_sdp/data/config.json) 
but we allow overriding those values with environment variables. 
If, at any point in the code, the config is queried (`config.get(key)`) 
for a key that isn't present in the top level of the JSON file, a 
warning is issued to add a default value to the JSON file. 
This helps ensure that we can track all possible configuration
values in the JSON config rather than having to bookkeep them elsewhere.
