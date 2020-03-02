# Copyright (C) 2020 IDIR Lab - UT Arlington
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License v3 as published by
#     the Free Software Foundation.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Contact Information:
#     See: https://idir.uta.edu/cli.html
#
#     Chengkai Li
#     Box 19015
#     Arlington, TX 76019
#

import numpy as np
from bert_adversarial.core.api.api_wrapper import ClaimSpotterAPI

if __name__ == '__main__':
    api = ClaimSpotterAPI()

    print('--- Batch Sentence Query ---')

    sentence_list = [
        'Donald Trump is the 45th President of the United States',
        'I really like cheese',
        'McDonalds earns $10 billion dollars each minute'
    ]

    print(sentence_list)
    print(api.batch_sentence_query(sentence_list))
    print('Test OK')

    print('--- Single Sentence Query ---')

    print(sentence_list[0])
    print(api.single_sentence_query(sentence_list[0]))
    print('Test OK')

    print('--- Command Line Query ---')

    while True:
        res = api.subscribe_cmdline_query()[0]
        idx = np.argmax(res)

        print(res)
        print('{} with probability {}'.format(api.return_strings[idx], res[idx]))
