import numpy as np
from api_wrapper import ClaimBusterAPI

if __name__ == '__main__':
    api = ClaimBusterAPI()

    while True:
        res = api.subscribe_query()
        idx = np.argmax(res, axis=1)

        print(res)

        print('{} with probability {}'.format(np.array(api.return_strings)[idx][0], res[0][idx][0]))
