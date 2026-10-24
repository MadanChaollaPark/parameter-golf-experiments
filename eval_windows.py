"""Causal evaluation windows with exact, non-overlapping target ownership.

Index t refers to target tokens[t+1], conditioned only on tokens[:t+1].
Weights are frozen. No optimizer, gradients, n-gram cache, or TTT is used.
"""
def windows(n_targets, context=1024, stride=64):
    if n_targets<1 or not 1<=stride<=context:raise ValueError('Invalid evaluation dimensions')
    scored=0
    while scored<n_targets:
        end=min(n_targets,scored+(context if scored==0 else stride))
        start=max(0,end-context)
        yield start,scored,end
        scored=end

def prove_coverage(n_targets,context=1024,stride=64):
    previous=0;count=0;windows_count=0
    for start,a,b in windows(n_targets,context,stride):
        assert a==previous and 0<=start<=a<b<=n_targets and b-start<=context
        # Every scored target at index t uses inputs start..t, strictly before target token t+1.
        count+=b-a;previous=b;windows_count+=1
    assert previous==n_targets and count==n_targets
    return {'targets':count,'windows':windows_count,'first_target_token_index':1,'last_target_token_index':n_targets,'exactly_once':True,'causal_by_index':True}

if __name__=='__main__':
    import json
    for length in [1,63,64,65,1023,1024,1025,2048,2061]:
        for stride in [1,64,256,1024]:
            seen=[0]*length
            for start,a,b in windows(length,1024,stride):
                for t in range(a,b):seen[t]+=1
            assert seen==[1]*length
            prove_coverage(length,1024,stride)
    print(json.dumps({str(s):prove_coverage(62021845,1024,s) for s in [64,1024]},indent=2))
