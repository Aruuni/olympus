"""Legacy script alias for the renamed td3 learner."""

import runpy


if __name__ == '__main__':
    runpy.run_module('olympus.algorithms.td3.learner',
                     run_name='__main__')
