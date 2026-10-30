import unittest
from tracking_demo.observation_diagnostics import exact_parents,signature,unique_index,person_at_json_index,identity_contributions,four_sets,stratified_sample,retain_review,condition_member_roles

def member(cam,obs):return {'frame':'0','camera':cam,'obs_id':obs}

class ObservationDiagnosticsTests(unittest.TestCase):
    def test_exact_parent_structures(self):
        b2={'a':[member('C1','a')],'b':[member('C2','b')]}
        b3={'x':[member('C1','a'),member('C3','c')],'y':[member('C2','b')],'z':[member('C3','d')]}
        self.assertEqual(exact_parents(b2,b3),{'x':('a','C3_joined'),'y':('b','no_C3'),'z':(None,'C3_new')})
    def test_duplicate_and_missing_fail(self):
        with self.assertRaises(ValueError):signature([member('C1','a')]*2)
        with self.assertRaises(ValueError):unique_index([{'a':1}]*2,lambda r:r['a'])
        with self.assertRaises(ValueError):exact_parents({'a':[member('C1','a')]},{'b':[member('C1','b')]})
        with self.assertRaises(ValueError):exact_parents({'a':[member('C1','a')]},{})
    def test_original_json_index_not_roi_index(self):
        records=[{'personID':999},{'personID':12},{'personID':344}]
        self.assertEqual(person_at_json_index(records,'2'),344)
    def test_four_sets_cover_both_missing(self):
        result=four_sets(range(5),{0,1},{1,2})
        self.assertEqual(result,{'both':{1},'B2_only':{0},'B3_only':{2},'neither':{3,4}})
        self.assertEqual(len(result['B3_only'])-len(result['B2_only']),0)
    def test_global_mapping_not_clear_or_nearest_dummy_boundary(self):
        mapping=[{'personID':'1','track_id':'9'},{'personID':'2','track_id':''},{'personID':'','track_id':'8'}]
        pairs=[{'frame':'0','personID':'1','track_id':'8','distance_m':'.1','valid_le_1m':'True'},
               {'frame':'0','personID':'1','track_id':'9','distance_m':'1','valid_le_1m':'True'},
               {'frame':'5','personID':'1','track_id':'9','distance_m':'1.000001','valid_le_1m':'False'}]
        result=identity_contributions(mapping,pairs)
        self.assertEqual(result,{(0,1):{'track_id':9,'distance_m':1.0}})
    def test_sampling_fixed_short_empty_no_replacement(self):
        config={'seed':20261017,'segment_order':['a','b'],'structure_order':['new','old'],'random_per_stratum':4}
        population=[{'segment':'a','structure':'new','frame':i,'B3_group':str(i)} for i in range(10)]+[{'segment':'b','structure':'old','frame':0,'B3_group':'x'}]
        a,strata=stratified_sample(population,config);b,_=stratified_sample(population[::-1],config)
        self.assertEqual(a,b);self.assertEqual(len(a),5);self.assertEqual(len({r['B3_group'] for r in a}),5)
        self.assertEqual([r['n_h'] for r in strata],[4,0,0,1]);self.assertEqual(a[-1]['inclusion_probability'],1)
    def test_uncertain_retained(self):
        samples=[{'case_id':'x','group':'g'}];reviews=[{'case_id':'x','semantic':'uncertain'}]
        result=retain_review(samples,reviews)
        self.assertEqual(result[0]['group'],'g');self.assertEqual(result[0]['review']['semantic'],'uncertain')
    def test_union_panel_keeps_condition_specific_members(self):
        # C24-like prior: B2 owns C1/C2; B3 owns only C3. No composite B3 claim.
        members={'B2':[member('C1','a'),member('C2','b')],'B3':[member('C3','c')]}
        self.assertEqual(condition_member_roles(members,'C1'),{'a':['B2']})
        self.assertEqual(condition_member_roles(members,'C3'),{'c':['B3']})
        members['B3'].append(member('C1','a'))
        self.assertEqual(condition_member_roles(members,'C1'),{'a':['B2','B3']})

if __name__=='__main__':unittest.main()
