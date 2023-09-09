from setuptools import find_packages,setup
    # find_packages will find all the packages in the ML application directory
from typing import List

HYPEN_EDOT = "-e ."

# meaning the function will take file_path(string) as an input and give a list as output.
def get_requirements(file_path:str)->List[str]:
    '''
    This Function will return the list of all the requirements(packages)
    '''
    requirements = []
    with open(file_path) as file_obj:   # open the requirements txt file and reading each line for packages
        requirements = file_obj.readlines()
        # whenever this is executed, \n from file will also be recorded.
        # using list comprehension to replace \n with blanks
        requirements = [req.replace('\n', '') for req in requirements]
        if HYPEN_EDOT in requirements:
            requirements.remove(HYPEN_EDOT)
    return requirements


# setup information about the application with the version and keep on updating it with new updates
setup(
    name = 'Student_Performance',
    version='0.0.1',
    author= 'Abhishek',
    author_email='vedanshtiwari.07@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')

    )


