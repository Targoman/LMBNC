##########################################################################################
# LMBNC: Language Model Based N-Gram Corrector
#
# Copyright 2020 by Targoman Co. Pjc.
#                                                                         
# LMBNC is free software: you can redistribute it and/or modify         
# it under the terms of the GNU Lesser General Public License as published 
# by the Free Software Foundation, either version 3 of the License, or     
# (at your option) any later version.                                      
#                                                                         
# LMBNC is distributed in the hope that it will be useful,              
# but WITHOUT ANY WARRANTY; without even the implied warranty of           
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            
# GNU Lesser General Public License for more details.                      
# You should have received a copy of the GNU Lesser General Public License 
# along with Targoman. If not, see <http://www.gnu.org/licenses/>.
#
##########################################################################################
# @author Zakieh Shakeri <z.shakeri@targoman.com>
# @author Behrooz Vedadian <vedadian@targoman.com>
##########################################################################################

#!/usr/bin/env python3

import sys
import argparse
import logging

from LMBNC import LMBNC

def main():

    parser = argparse.ArgumentParser('LMBNC')
    parser.add_argument(
        '-i',
        '--input_corpus',
        help='Path to input corpus for correction.',
        required=True
    )
    parser.add_argument(
        '-l',
        '--lm_model_path',
        help='Path to language model file path.',
        required=True
    )
    parser.add_argument(
        '-o',
        '--output_corpus',
        help='Path to output corpus for correction.',
        required=True
    )

    args = parser.parse_args()

    logger = logging.getLogger('LMBNC')
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s %(message)s')
    logger.setFormatter(formatter)

    logger.info('Loading language model')
    lmbnc = LMBNC(args.lm_model_path)
    logger.info('Loading corpus')
    lmbnc.load_corpus(args.input_corpus)
    logger.info('Correcting N-Grams')
    lmbnc.correct_ngrams()
    logger.info('Saving output')
    lmbnc.save_corpus(args.output_corpus)
    logger.info('Done')

if __name__ == "__main__":
    main()