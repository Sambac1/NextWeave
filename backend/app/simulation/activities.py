"""Read-only activity observation. Public settlements only; no target rewriting."""
import re
from app.domain.activities import ActivityAnnotation

VERSION = 'activity-observations-1'
FAMILIES = {'proposal':'Proposals and requests','negotiation':'Communication and negotiation','agreement':'Decisions and commitments',
            'communication':'Announcements','implementation':'Implementation','resource':'Resource allocation','outcome':'Outcome changes'}
STAGES = {'proposed':'Proposal or request','discussed':'Discussion or negotiation','conditional':'Conditional acceptance',
          'accepted':'Acceptance','rejected':'Rejection','announced':'Public announcement',
          'started':'Action initiated','ongoing':'Action in progress','completed':'Completed',
          'failed':'Failure or breakdown','paused':'Paused','cancelled':'Cancelled or terminated'}
PATTERNS = [
 ('proposal','proposed',r'\b(?:proposed|offered|requested|submitted|counteroffer|called on|called for)\b|提出|提议|请求|提交'),
 ('negotiation','discussed',r'\b(?:discussed|negotiated|conveyed|relayed|exchanged|consulted|transmitted|responded)\b|协商|磋商|商讨|沟通|传达|转交|交换'),
 ('agreement','rejected',r'\b(?:rejected|refused|declined)\b|拒绝|否决'),
 ('agreement','accepted',r'\b(?:accepted|accepts|agreed|approved|signed)\b|接受|同意|批准|签署'),
 ('communication','announced',r'\b(?:announced|published|disclosed|issued a statement)\b|宣布|公布|发布声明'),
 ('implementation','started',r'\b(?:began|started|initiated|commenced)\b|开始|启动|生效'),
 ('implementation','ongoing',r'\b(?:conducted|carried out|implemented|executed|continues?|continued|sustained|maintained|remains? in (?:effect|force))\b|实施|执行|开展|继续|维持|仍然有效|仍有效|仍在生效|依然有效'),
 ('resource','ongoing',r'\b(?:allocated|invested|reinforced|deployed|delivered|reallocated)\b|调配|投入|拨付|增配|交付'),
 ('outcome','completed',r'\b(?:completed|finished|achieved)\b|完成|达成'),
 ('outcome','failed',r'\b(?:failed|collapsed|broke down)\b|失败|破裂'),
 ('implementation','paused',r'\b(?:paused|suspended|halted)\b|暂停|中止|暂缓执行'),
 ('implementation','cancelled',r'\b(?:cancelled|canceled|terminated)\b|取消|终止|撤销'),
]
NEGATION = re.compile(r'\b(?:not|never|no longer|didn.t|cannot|unable to)\b|没有|并未|尚未|未能|不再|不能|未|不',re.I)
FUTURE = re.compile(r'\b(?:will|would|could|may|might|if|before any|in exchange for|offered to|proposed to|plan(?:ned)? to|agreed to|seeking to|promised|intends to|expects to)\b|计划|拟|将会|若|如果|以便|旨在|提出|提议|承诺|要求|希望|寻求|愿意|愿以|考虑|打算|可能|将',re.I)

def _conditional(text):
    return bool(re.search(r'\b(?:conditional|in principle|on .*terms|conditioned on|provided that|subject to|only if|in exchange for)\b|附条件|原则上|有条件',text,re.I))

def observations(event):
    text=str(event.get('description') or '')
    results=[]; seen=set()
    annotations=event.get('activity_annotations') or []
    for raw in annotations:
        try: item=ActivityAnnotation.model_validate(raw)
        except (ValueError,TypeError): continue
        if item.quote not in text: continue
        if item.stage in {'completed','started','ongoing','accepted'} and (NEGATION.search(item.quote) or FUTURE.search(item.quote)):
            continue
        if item.stage=='accepted' and _conditional(item.quote):
            item.stage='conditional'
        key=(item.family,item.stage,item.quote)
        if key not in seen:
            seen.add(key);results.append(item.model_dump())
    # Old events often bundle several behaviors. Preserve quoted clauses without
    # treating an action proposal or hypothetical consequence as an execution.
    for clause in re.split(r'(?<=[.!?。；;])\s*|,\s+(?:while|but|and)\s+',text):
        if not 8 <= len(clause.strip()) <= 1800: continue
        clause=clause.strip()
        for family,stage,pattern in PATTERNS:
            match=re.search(pattern,clause,re.I)
            if not match: continue
            prefix=clause[max(0,match.start()-80):match.start()]
            if stage=='accepted' and re.search(r'\b(?:an?|the|mutually|previously|widely)\s+$',prefix,re.I): continue
            if NEGATION.search(prefix[-40:]) or FUTURE.search(prefix): continue
            if stage=='accepted' and _conditional(clause): stage='conditional'
            # Prefer model annotations for an already-labelled clause/category.
            if any(r['family']==family and r['stage']==stage and (r['quote'] in clause or clause in r['quote']) for r in results): continue
            key=(family,stage,clause)
            if key not in seen:
                seen.add(key);results.append({'family':family,'stage':stage,'quote':clause,'method':'legacy_structure_rule'})
    return results


def annotate_events(events):
    """Populate anchored observations after settlement, with no model request."""
    for event in events:
        valid=observations(event.model_dump())
        event.activity_annotations=[ActivityAnnotation.model_validate({k:r[k] for k in ('family','stage','quote','method')}) for r in valid[:8]]


def activity_frequencies(records, planned):
    completed=[r for r in records if r.get('status')=='completed']
    groups={};unlabelled=0;total=0
    for sample in completed:
        seen_events=set()
        for frame in sample.get('rounds',[]):
            for event in frame.get('timeline',[]):
                identifier=event.get('timeline_id')
                if not identifier or identifier in seen_events: continue
                seen_events.add(identifier);total+=1
                found=observations(event)
                if not found: unlabelled+=1
                for row in found:
                    key=f"{row['family']}:{row['stage']}"
                    group=groups.setdefault(key,{'id':key,'family':row['family'],'family_label':FAMILIES[row['family']],
                        'stage':row['stage'],'label':STAGES[row['stage']],'sample_ids':set(),'evidence':[],'evidence_count':0,'_seen':set()})
                    marker=(sample['sample_id'],identifier,row['quote'])
                    if marker in group['_seen']: continue
                    group['_seen'].add(marker);group['sample_ids'].add(sample['sample_id'])
                    group['evidence_count']+=1
                    if len(group['evidence']) >= 32: continue
                    group['evidence'].append({'sample_id':sample['sample_id'],'step':frame.get('step'),
                        'timeline_id':identifier,'quote':row['quote'],'method':row['method'],'actors':event.get('actors',[])})
    output=[]
    for group in groups.values():
        group.pop('_seen');ids=sorted(group['sample_ids']);group['sample_ids']=ids
        group.update(count=len(ids),completed_samples=len(completed),frequency=len(ids)/len(completed) if completed else None)
        output.append(group)
    output.sort(key=lambda g:(list(FAMILIES).index(g['family']),list(STAGES).index(g['stage'])))
    return {'version':VERSION,'basis':'posthoc_public_activity_observation','planned_samples':planned,
        'completed_samples':len(completed),'unfinished_samples':max(0,planned-len(completed)),
        'public_events':total,'unlabelled_events':unlabelled,'groups':output}
