"""Synthetic counterexample using the unmodified PR2135 C helper, no real validation."""
def check(source,output):
 import pathlib,subprocess,ctypes,json,ast,types
 source=pathlib.Path(source);output=pathlib.Path(output)
 libpath=output.parent/'ngram-test.so'
 subprocess.run(['gcc','-O2','-shared','-fPIC','-o',str(libpath),str(source/'online_ngram_state.c')],check=True)
 lib=ctypes.CDLL(str(libpath));lib.online_ngram_state_create.restype=ctypes.c_void_p;lib.online_ngram_state_create.argtypes=[ctypes.c_int]*3
 lib.online_ngram_state_seed_prefix_token.argtypes=[ctypes.c_void_p,ctypes.c_uint16];lib.online_ngram_state_destroy.argtypes=[ctypes.c_void_p]
 u16=ctypes.POINTER(ctypes.c_uint16);u8=ctypes.POINTER(ctypes.c_uint8);f32=ctypes.POINTER(ctypes.c_float)
 lib.online_ngram_state_process_chunk.argtypes=[ctypes.c_void_p,u16,ctypes.c_int64,u8,u8,u16,f32,u16,f32,u8]
 def hints(tokens):
  state=lib.online_ngram_state_create(15,12,12)
  # Match build_hints_for_targets: seed with first target, then process targets.
  lib.online_ngram_state_seed_prefix_token(state,tokens[0])
  n=len(tokens);x=(ctypes.c_uint16*n)(*tokens);new=(ctypes.c_uint8*8192)();boundary=(ctypes.c_uint8*8192)();boundary[1]=1
  a=(ctypes.c_uint16*n)();b=(ctypes.c_float*n)();c=(ctypes.c_uint16*n)();d=(ctypes.c_float*n)();e=(ctypes.c_uint8*n)()
  assert lib.online_ngram_state_process_chunk(state,x,n,new,boundary,a,b,c,d,e)==0
  result=list(zip(a,b));lib.online_ngram_state_destroy(state);return result
 ctx=list(range(2,17));A=ctx+[99];B=ctx+[88]+[100]*20
 prefix=A+[1,200]*63+[1]
 full=hints(prefix+B);alone=hints(B);i=len(prefix)+len(ctx)
 tree=ast.parse((source/'train_gpt.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_build_ttt_global_batches');ns={}
 exec(compile(ast.Module(body=[fn],type_ignores=[]),'<native-queue>','exec'),ns)
 lengths=[len(A)+1]+[2]*63+[len(B)+1];offset=0;entries=[]
 for index,length in enumerate(lengths):entries.append((index,(offset,length)));offset+=length
 queue=ns['_build_ttt_global_batches'](entries,types.SimpleNamespace(ttt_batch_size=64))
 assert queue[0][1][0][0]==64
 r={'synthetic_only':True,'native_C_source_unmodified':True,'token_order':16,'A_doc_length_including_BOS':len(A)+1,'B_doc_length_including_BOS':len(B)+1,'B_target_hint_with_canonical_A_prefix':full[i],'B_target_hint_without_A':alone[len(ctx)],'native_descending_length_queue_scores_B_before_A':True,'synthetic_document_count':65,'first_native_queue_document_indices':[i for i,_ in queue[0][1]],'cross_document_dependency_demonstrated':full[i]!=alone[len(ctx)],'interpretation':'Global canonical-order hints retain A content although native descending-length TTT queue scores B first. This conflicts with requested document independence and actual score-before-update order. No claim about numerical impact on the published benchmark.'}
 assert r['cross_document_dependency_demonstrated'];output.write_text(json.dumps(r,indent=2)+'\n');return r
