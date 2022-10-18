'''
Author: Sean Cowan
Purpose: MSc Thesis
Date Created: 17-10-2022

This module performs unit tests for the mga_low_thrust_utilities file.
'''

# General imports
import sys
sys.path.append('../')
from tudatpy.kernel import constants
import mga_ltto.mga_low_thrust_utilities as util
import unittest as unit
import numpy as np

###########################################################################
# General parameters ######################################################
###########################################################################

write_results_to_file = False
julian_day = constants.JULIAN_DAY
seed = 421

####################################################################
# MGSO Unit tests ##################################################
####################################################################

class unitTestsUtilities(unit.TestCase):

    """
    Unit tests for the transfer_body_order_conversion class
    """

    def test_get_transfer_body_integers_strip(self):

        transfer_body_order = ["Venus", "Venus", "Earth", "Mercury", "Null", "Null", "Null", "Null"]
        expected_output = np.array([2, 2, 3, 5])
        output = \
        util.transfer_body_order_conversion.get_transfer_body_integers(transfer_body_order,
                strip=True)
        self.assertEqual(output.all(), expected_output.all())

    def test_get_transfer_body_integers_nostrip(self):

        transfer_body_order = ["Venus", "Venus", "Earth", "Mercury", "Null", "Null", "Null", "Null"]
        expected_output = np.array([2, 2, 3, 5, 0, 0, 0, 0])
        output = \
        util.transfer_body_order_conversion.get_transfer_body_integers(transfer_body_order,
                strip=False)
        self.assertEqual(output.all(), expected_output.all())

    def test_get_transfer_body_list_strip(self):

        transfer_body_integers = np.array([2, 2, 3, 5, 0, 0, 0, 0])
        expected_output = ["Venus", "Venus", "Earth", "Jupiter"]
        output = \
        util.transfer_body_order_conversion.get_transfer_body_list(transfer_body_integers,
                strip=True)
        self.assertEqual(output, expected_output)

    def test_get_transfer_body_list_nostrip(self):

        transfer_body_integers = np.array([2, 2, 3, 5, 0, 0, 0, 0])
        expected_output = ["Venus", "Venus", "Earth", "Jupiter", "Null", "Null", "Null", "Null"]
        output = \
        util.transfer_body_order_conversion.get_transfer_body_list(transfer_body_integers,
                strip=False)
        self.assertEqual(output, expected_output)

    def test_get_mga_characters_from_list(self):
        transfer_body_list = ["Venus", "Venus", "Earth", "Jupiter"]
        expected_output = 'VVEJ'
        output = \
        util.transfer_body_order_conversion.get_mga_characters_from_list(transfer_body_list)
        self.assertEqual(output, expected_output)

    def test_get_mga_list_from_characters(self):
        mga_characters = 'VVEJ'
        expected_output = ["Venus", "Venus", "Earth", "Jupiter"]
        output = \
        util.transfer_body_order_conversion.get_mga_list_from_characters(mga_characters)
        self.assertEqual(output, expected_output)

    def test_get_list_of_legs_from_chars(self):
        mga_characters = 'EMEMJ'
        expected_output = ['EM', 'ME', 'EM', 'MJ']
        output = \
        util.transfer_body_order_conversion.get_list_of_legs_from_characters(mga_characters)
        self.assertEqual(output, expected_output)

    def test_get_dict_of_legs_from_characters(self):
        mga_characters = 'EMEMJ'
        expected_output = {'EM' : 0, 'ME' : 1, 'EM' : 2, 'MJ' : 3}
        output = \
        util.transfer_body_order_conversion.get_dict_of_legs_from_characters(mga_characters)
        self.assertEqual(output, expected_output)

    """
    Unit tests for the other functions in utilities
    """

    # def test_get_low_thrust_transfer_object(self):

    # def test_get_node_times(self):


if __name__ == '__main__':
    unit.main()

