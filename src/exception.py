"""
Building own custom exception for the project

"""
# This module provides access to some variables used or maintained by the interpreter and
# to functions that interact strongly with the interpreter. It is always available.
import sys   # any exception handling going on sys will already know of it
import logging
def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()  # carrys 3 values, 3rd imp -> info of on which file, which line
    file_name = exc_tb.tb_frame.f_code.co_filename # GEtting where filename is stored(in documentation)
    error_message = f'Error encountered under python Script name [{file_name}] line number [{exc_tb.tb_lineno}] error message[{str(error)}]'
    return error_message


class CustomException(Exception): # Inheriting from Parent Exception
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message) # inheriting __init__ function
        self.error_message  = error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message


'''
# CHECKING PROGRAM
if __name__ == '__main__':

    try:
        a = 1/0
    except Exception as e:
        logging.info('Divide by Zero Error')
        raise CustomException(e,sys)

'''