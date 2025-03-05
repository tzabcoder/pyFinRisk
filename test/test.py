# Local Imports
from test.UnitTest import UnitTest

def main():
    """
    * main()
    *
    * Runs the unit test procedures for the RiskEngine library.
    """

    # Create and run the UT procedures
    test_procedure = UnitTest()
    test_procedure.run()

if __name__ == "__main__":
    main()