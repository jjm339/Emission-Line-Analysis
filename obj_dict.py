"""
Dirctionary of Observations to Objects
"""
obj_dict = {'0050':'NGC6278',
            '0051':'NGC6278',
            '0052':'NGC6278',
            '0057':'Sgr2',
            '0058':'Sgr2',
            '0059':'Sgr2',
            '0061':'Sgr2',
            '0073':'Hyi1',
            '0074':'Hyi1',
            '0076':'Hyi1',
            '0077':'Hyi1',
            '0321':'Ret2',
            '0322':'Ret2',
            '0323':'Ret2',
            '0324':'Ret2',
            '0908':'HD161817 B8-09',
            '0909':'Hd161817 R7-03',
            '0914':'Sgr2',
            '0915':'Sgr2',
            '0918':'Sgr2',
            '0919':'Sgr2',
            '0932':'Hyi1',
            '0933':'Hyi1',
            '0935':'Hyi1',
            '0936':'Hyi1',
            '0939':'HD6268 R7-15',
            '0940':'HD6268 B5-05',
            '2782':'Ret2',
            '2783':'Ret2',
            '2785':'Ret2',
            '2786':'Ret2',
            '2789':'Ret2',
            '2791':'HD21581 R5-15',
            '2792':'HD21581 B3-03'}

"""
Define functions
"""
def get_obj(obs, objects):
    for key in objects:
        if key in obs:
            return objects[key]
    # Fails to find the observation number, returns None.
    return None