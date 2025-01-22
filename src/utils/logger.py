import logging

logging.basicConfig(
     level=logging.INFO,
     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d'
)

def log_preprocess_info(method, status):
     ''' status: start; finish '''
     logging.info(f"Function {method} is {status}")