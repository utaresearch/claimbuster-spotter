import numpy as np
from api_wrapper import ClaimBusterAPI

if __name__ == '__main__':
    api = ClaimBusterAPI()

    while True:
        res = api.subscribe_cmdline_query()
        idx = np.argmax(res)

        print(res)

        print('{} with probability {}'.format(api.return_strings[idx], res[idx]))
