import torch

def left_neighbors(sides,  size):
    neighbors = ['back', 'front', 'top', 'down']
    l, r, t, d = [sides[n] for n in neighbors]    
    return [
        l[:size, :, :],
        r[:size, :, :],
        t[:, :size, :],
        d[:, :size, :],
    ]

def right_neighbors(sides,  size):
    neighbors = ['front', 'back', 'top', 'down']
    l, r, t, d = [sides[n] for n in neighbors]    
    return [
        l[-size:, :, :],
        r[-size:, :, :],
        t[:, -size:, :],
        d[:, -size:, :],
    ]

def front_neighbors(sides,  size):
    neighbors = ['left', 'right', 'top', 'down']
    l, r, t, d = [sides[n] for n in neighbors]    
    return [
        l[-size:, :, :],
        r[-size:, :, :],
        t[-size:, :, :],
        d[-size:, :, :],
    ]

def back_neighbors(sides,  size):
    neighbors = ['right', 'left', 'top', 'down']
    l, r, t, d = [sides[n] for n in neighbors]    
    return [
        l[:size, :, :],
        r[:size, :, :],
        t[:size, :, :],
        d[:size, :, :],
    ]

def top_neighbors(sides,  size):
    neighbors = ['left', 'right', 'back', 'front']
    l, r, t, d = [sides[n] for n in neighbors]    
    return [
        l[:, -size:, :],
        r[:, -size:, :],
        t[:, -size:, :],
        d[:, -size:, :],
    ]

def down_neighbors(sides,  size):
    neighbors = ['left', 'right', 'front', 'back']
    l, r, t, d = [sides[n] for n in neighbors]    
    return [
        l[:, :size, :],
        r[:, :size, :],
        t[:, :size, :],
        d[:, :size, :],        
    ]


def get_neighbors(sides,  side_name, size):
    if side_name == 'left':
        return left_neighbors(sides,  size)
    elif side_name == 'right':
        return right_neighbors(sides,  size)
    elif side_name == 'front':
        return front_neighbors(sides,  size)
    elif side_name == 'back':
        return back_neighbors(sides,  size)
    elif side_name == 'top':
        return top_neighbors(sides,  size)
    elif side_name == 'down':
        return down_neighbors(sides,  size)
    raise Exception(f'Unknown side name {side_name}')
    
def make_tris(size, device):
    tl = torch.triu(torch.ones(size, size)) - (torch.eye(size) * 0.5)
    dr = torch.tril(torch.ones(size, size)) - (torch.eye(size) * 0.5)    
    tl, dr = tl.to(device), dr.to(device)
    return {
        'tl': tl[:,:,None],
        'lt': tl.t().clone()[:,:,None],        
        'tr': tl.t().flip(dims=(0,)).t()[:,:,None],
        'rt': tl.flip(dims=(0,)).t()[:,:,None],
        'dr': dr.t()[:,:,None],
        'rd': dr[:,:,None],
        'ld': dr.flip(dims=(0,)).t()[:,:,None],
        'dl': dr.t().flip(dims=(0,)).t()[:,:,None],      
    }

def front_corners(size, l, r, t, d):
    tris = make_tris(size, l.device)
    lt = l[:, -size:, :]
    tl = t[:, :size, :] 

    tr = t[:, -size:, :]
    rt = r[:, -size:, :] 

    ld = l[:, :size, :] 
    dl = d[:, :size, :]

    rd = r[:, :size, :]
    dr = d[:, -size:, :]

    ltc = lt * tris['lt'] + tl * tris['tl']
    trc = tr * tris['tr'] + rt * tris['rt']
    ldc = ld * tris['ld'] + dl * tris['dl'] 
    drc = dr * tris['rd'] + rd * tris['dr']
    
    return ltc, trc, ldc, drc

def back_corners(size, l, r, t, d):
    tris = make_tris(size, l.device)
    lt = l[:, -size:, :]
    tl = t[:, -size:, :] 

    tr = t[:, :size, :]
    rt = r[:, -size:, :] 

    ld = l[:, :size, :] 
    dl = d[:, -size:, :]

    rd = r[:, :size, :]
    dr = d[:, :size, :]

    ltc = lt * tris['lt'] + tl * tris['tl']
    trc = tr * tris['tr'] + rt * tris['rt']
    ldc = ld * tris['ld'] + dl * tris['dl'] 
    drc = dr * tris['rd'] + rd * tris['dr']
    
    return ltc, trc, ldc, drc

def left_corners(size, l, r, t, d):
    tris = make_tris(size, l.device)
    lt = l[:, -size:, :]
    tl = t[:size, :,  :] 

    ld = l[:, :size, :] 
    dl = d[:size, :, :]

    tr = t[-size:,:,  :]
    rt = r[:,-size:, :] 

    rd = r[:, :size, :]
    dr = d[-size:,:,  :]

    ltc = lt * tris['lt'] + tl * tris['tl']
    trc = tr * tris['tr'] + rt * tris['rt']
    ldc = ld * tris['ld'] + dl * tris['dl'] 
    drc = dr * tris['rd'] + rd * tris['dr']    
    return ltc, trc, ldc, drc

def right_corners(size, l, r, t, d):
    tris = make_tris(size, l.device)
    lt = l[:, -size:, :]
    tl = t[-size:, :,  :] 

    ld = l[:, :size, :] 
    dl = d[-size:, :, :]

    tr = t[:size,:,  :]
    rt = r[:,-size:, :] 
    
    rd = r[:, :size, :]
    dr = d[:size,:,  :]
    
    ltc = lt * tris['lt'] + tl * tris['tl']
    trc = tr * tris['tr'] + rt * tris['rt']
    ldc = ld * tris['ld'] + dl * tris['dl'] 
    drc = dr * tris['rd'] + rd * tris['dr']    
    return ltc, trc, ldc, drc

def top_corners(size, l, r, t, d):
    tris = make_tris(size, l.device)
    lt = l[:size,:,  :]
    tl = t[:size, :,  :] 
    
    ld = l[-size:,:,  :] 
    dl = d[:size, :, :]

    tr = t[-size:, :,  :]
    rt = r[:size,:, :] 

    rd = r[-size:,:,  :]
    dr = d[-size:,:,  :]
    
    ltc = lt * tris['lt'] + tl * tris['tl']
    trc = tr * tris['tr'] + rt * tris['rt']
    ldc = ld * tris['ld'] + dl * tris['dl'] 
    drc = dr * tris['rd'] + rd * tris['dr']
    return ltc, trc, ldc, drc

def down_corners(size, l, r, t, d):
    tris = make_tris(size, l.device)
    lt = l[-size:,:,  :]
    tl = t[:size, :,  :] 

    ld = l[:size,:,  :] 
    dl = d[:size, :, :]

    tr = t[-size:, :,  :]
    rt = r[-size:,:, :] 

    rd = r[:size,:,  :]
    dr = d[-size:,:,  :]
    
    ltc = lt * tris['lt'] + tl * tris['tl']
    trc = tr * tris['tr'] + rt * tris['rt']
    ldc = ld * tris['ld'] + dl * tris['dl'] 
    drc = dr * tris['rd'] + rd * tris['dr']    
    return ltc, trc, ldc, drc


def get_corners(size, side_name, l, r, t, d):
    if side_name == 'front':
        return front_corners(size, l, r, t, d)
    elif side_name == 'back':
        return back_corners(size, l, r, t, d)
    elif side_name == 'left':
        return left_corners(size, l, r, t, d)
    elif side_name == 'right':
        return right_corners(size, l, r, t, d)
    elif side_name == 'top':
        return top_corners(size, l, r, t, d)
    elif side_name == 'down':
        return down_corners(size, l, r, t, d)
    raise Exception(f'Unknown side name {side_name}')
    

def pad_left_right(sides, side_name, kernel_size):
    o = sides[side_name]
    size = (kernel_size - 1) // 2
    l, r, t, d = get_neighbors(sides, side_name, size)
    lt, tr, ld, dr = get_corners(size, side_name, l, r, t, d)

    top = torch.cat((lt, t, tr), dim=0)
    down = torch.cat((ld, d, dr), dim=0)
    middle = torch.cat((l, o, r), dim=0)
    return torch.cat((top, middle, down), dim=1)

def pad_front_back(sides, side_name, kernel_size):
    o = sides[side_name]
    size = (kernel_size - 1) // 2
    l, r, t, d = get_neighbors(sides, side_name, size)
    lt, tr, ld, dr = get_corners(size, side_name, l, r, t, d)
    
    top = torch.cat((lt, t.permute(1, 0, 2), tr), dim=0)
    down = torch.cat((ld, d.permute(1, 0, 2), dr), dim=0)
    middle = torch.cat((l, o, r), dim=0)
    return torch.cat((top, middle, down), dim=1)

def pad_top_down(sides, side_name, kernel_size):
    o = sides[side_name]
    size = (kernel_size - 1) // 2
    l, r, t, d = get_neighbors(sides, side_name, size)
    lt, tr, ld, dr = get_corners(size, side_name, l, r, t, d)
    
    top = torch.cat((lt, t, tr), dim=0)
    down = torch.cat((ld, d, dr), dim=0)
    middle = torch.cat((l.permute(1, 0, 2), o, r.permute(1, 0, 2)), dim=0)
    return torch.cat((top, middle, down), dim=1)

def pad_side(sides, side_name, kernel_size):
    if side_name in ['left', 'right']:
        return pad_left_right(sides, side_name, kernel_size)
    elif side_name in ['front', 'back']:
        return pad_front_back(sides, side_name, kernel_size)
    elif side_name in ['top', 'down']:
        return pad_top_down(sides, side_name, kernel_size)
    raise Exception(f'Unknown side name {side_name}')