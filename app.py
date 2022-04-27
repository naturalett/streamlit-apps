from sqlalchemy import create_engine
import pandas as pd
import hashlib
import statistics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go

import random
import streamlit as st

engine = create_engine('postgresql://redshift-prod.naturalint.com:5439/warehouse?')

st.set_page_config(layout='wide')

st.title('AA Test over Time')

## get the data for the filters
@st.cache(ttl=86400)
def get_filters():
    print('getting dims')
    engine = create_engine('postgresql://redshift-prod.naturalint.com:5439/warehouse?')
    dim_site = f"""select distinct site_name from dim_sites"""
    df_sites =  pd.read_sql_query(dim_site, engine)

    dim_vertical = f"""select distinct vertical_name from dim_verticals"""
    df_verticals =  pd.read_sql_query(dim_vertical, engine)

    dim_segments = f"""select distinct segment_name from dim_segments"""
    df_segments =  pd.read_sql_query(dim_segments, engine)

    dim_ts = f"""select traffic_source_name
                 from v_funnel_facts_analysts
                 where unified_date>=getdate()-30
                 group by 1
                 having count(distinct visit_iid)>100"""
    df_ts = pd.read_sql_query(dim_ts, engine)

    dim_lp = f"""select landing_page_uri
                 from v_funnel_facts_analysts
                 where unified_date>=getdate()-30
                 group by 1
                 having count(distinct visit_iid)>100"""
    df_lp = pd.read_sql_query(dim_lp, engine)

    return df_sites, df_verticals, df_segments, df_lp, df_ts

a, b, c, d, e = get_filters()

num_of_aa_iterations = 1000

#a

with st.form(key='my_form'):
    c1, c2, c3 = st.columns(3)
    with c1:
        #num_of_aa_iterations = int(st.number_input('Number of Iterations', min_value=1, step=1, value=1000))
        start_date = st.date_input('Start Date')
        end_date = st.date_input('End Date')
        site = st.multiselect(label='Site', options=a) #st.text_input('Site')
    with c2:
        vertical = st.multiselect(label='Vertical', options=b) #st.text_input('vertical')
        segment = st.multiselect(label='Segment', options=c)
        platform = st.multiselect(label='Platform', options=['Desktop', 'Mobile', 'Tablet'])
    with c3:
        channel = st.multiselect(label='Channel', options=['Paid Search', 'Paid Social'])
        lp = st.multiselect(label='Landing Page', options=d)
        ts = st.multiselect(label='Traffic Source', options=e)
    submit_button = st.form_submit_button(label='Submit')

## parameters to determine when the interval is stable
ctr_parameter = 0.15
cr_parameter = 0.5
epv_parameter = 0.5

## helping variables for cases where there's no value or one or more values
if len(vertical)>0:
    vertical3 = 'yes'
    vertical2 = str(vertical)[2:-2]
else:
    vertical3 = 'no'
    vertical2 = str(vertical)[1:-1]

if len(platform)>0:
    platform3 = 'yes'
    platform2 = str(platform)[2:-2]
else:
    platform3 = 'no'
    platform2 = str(platform)[1:-1]

if len(site)>0:
    site3 = 'yes'
    site2 = str(site)[2:-2]
else:
    site3 = 'no'
    site2 = str(site)[1:-1]

if len(channel)>0:
    channel3 = 'yes'
    channel2 = str(channel)[2:-2]
else:
    channel3 = 'no'
    channel2 = str(channel)[1:-1]

if len(segment)>0:
    segment3 = 'yes'
    segment2 = str(segment)[2:-2]
else:
    segment3 = 'no'
    segment2 = str(segment)[1:-1]

if len(lp)>0:
    lp3 = 'yes'
    lp2 = str(lp)[2:-2]
else:
    lp3 = 'no'
    lp2 = str(lp)[1:-1]

if len(ts)>0:
    ts3 = 'yes'
    ts2 = str(ts)[2:-2]
else:
    ts3 = 'no'
    ts2 = str(ts)[1:-1]


#q = f"""select cast ("{vertical}" as varchar(1000))"""

#qu = f"""select * from dim_verticals where (case when '{vertical3}'='no' then true else vertical_name in ('{vertical2}') end)"""

## Query on RS to extract user_ids
query = f"""select user_id, unified_date, 
    count(distinct visit_iid) as visits, count(distinct visit_iid_product) as uclicks, count(distinct visit_iid_cid_exist) as clickers, sum(estimated_conversions) as cons, sum(estimated_earnings_usd) as rev
    from v_funnel_facts_analysts
    where 1=1
    and (case when '{site3}'='no' then true else site_name in ('{site2}') end) 
    and (case when '{vertical3}'='no' then true else landing_vertical_name in ('{vertical2}') end)
    and (case when '{platform3}'='no' then true else agent_platform in ('{platform2}') end)
    and traffic_type='users'
    and (case when '{channel3}'='no' then true else traffic_join in ('{channel2}') end)
    and (case when '{segment3}'='no' then true else landing_segment_name in ('{segment2}') end)
    and (case when '{lp3}'='no' then true else landing_page_uri in ('{lp2}') end)
    and (case when '{ts3}'='no' then true else traffic_source_name in ('{ts2}') end)
    and lower(landing_page_type_name)='chart'
    and unified_date between '{start_date}' and '{end_date}'

    group by 1,2"""

df = pd.read_sql_query(query, engine)
 
date_list = df['unified_date'].drop_duplicates().sort_values().to_list()

## define functions for calculating KPIs
def get_uclicks(df):
    uclicks_a = df['uclicks'].iloc[:len(df)//2].sum()
    uclicks_b = df['uclicks'].iloc[len(df)//2:].sum()
    return uclicks_a, uclicks_b

def get_visits(df):
    visits_a = df['visits'].iloc[:len(df)//2].sum()
    visits_b = df['visits'].iloc[len(df)//2:].sum()
    return visits_a, visits_b

def get_cons(df):
    cons_a = df['cons'].iloc[:len(df)//2].sum()
    cons_b = df['cons'].iloc[len(df)//2:].sum()
    return cons_a, cons_b

def get_rev(df):
    rev_a = df['rev'].iloc[:len(df)//2].sum()
    rev_b = df['rev'].iloc[len(df)//2:].sum()
    return rev_a, rev_b

def get_ctr(uclicks, visits):
    ctr_a = uclicks[0] / visits[0]
    ctr_b = uclicks[1] / visits[1]
    return ctr_a, ctr_b
            
def get_cr(uclicks, cons):
    cr_a = cons[0] / uclicks[0]
    cr_b = cons[1] / uclicks[1]
    return cr_a, cr_b

def get_epv(visits, rev):
    epv_a = rev[0] / visits[0]
    epv_b = rev[1] / visits[1]
    return epv_a, epv_b

def merge_results(old_results, new_results):
    merged = {}
    for key in old_results.keys():
        merged[key] = old_results[key] + new_results[key]
    return merged

def get_lifts(result):
    lifts = {}
    ctr_a, ctr_b = get_ctr(result["uclicks"], result["visits"])
    lifts["ctr_lift"] = ctr_b/ctr_a-1
    
    cr_a, cr_b = get_cr(result["uclicks"], result["cons"])
    lifts["cr_lift"] = cr_b/cr_a-1
    
    epv_a, epv_b = get_epv(result["visits"], result["rev"])
    lifts["epv_lift"] = epv_b/epv_a-1
    
    return lifts

def generate_iteration_results(df, prev_results):
    s_df = df.reindex(np.random.permutation(df.index))
    
    result = {}
    result["uclicks"] = np.array(get_uclicks(s_df))
    result["visits"] = np.array(get_visits(s_df))
    result["cons"] = np.array(get_cons(s_df))
    result["rev"] = np.array(get_rev(s_df))
    
    merged_result = merge_results(prev_results, result)
    return get_lifts(merged_result), merged_result


## iterate the data for the AA results
average_ctrs = []
stdv_ctrs = []
ctrs_min_bound = []
ctrs_max_bound = []
average_crs = []
stdv_crs = []
crs_min_bound = []
crs_max_bound = []
average_epvs = []
stdv_epvs = []
epvs_min_bound = []
epvs_max_bound = []


prev_results_list = [{"uclicks" : np.array([0, 0]), 
                      "visits" : np.array([0, 0]), 
                      "cons" : np.array([0, 0]), 
                      "rev" : np.array([0, 0])} for i in range(num_of_aa_iterations)]

for date in date_list:
    print(f'We have started the AA iteration for the date range that ends in {date}')
    new_df = df[['user_id', 'unified_date', 'visits', 'uclicks', 'clickers', 'cons', 'rev']][df['unified_date']==date]
    
    random.shuffle(prev_results_list)
    
    lift_list = []
    for iteration in range(num_of_aa_iterations):
        lifts, merged_result = generate_iteration_results(new_df, prev_results_list[iteration])
        prev_results_list[iteration] = merged_result
        lift_list.append(lifts)
    
    ctr_list = [l["ctr_lift"] for l in lift_list]
    cr_list = [l["cr_lift"] for l in lift_list]
    epv_list = [l["epv_lift"] for l in lift_list]
    
    average_ctr = statistics.mean(ctr_list)
    average_ctrs.append(round(average_ctr*100,2))
    
    stdv_ctr = statistics.stdev(ctr_list)
    stdv_ctrs.append(stdv_ctr)

    average_cr = statistics.mean(cr_list)
    average_crs.append(round(average_cr*100,2))
    
    stdv_cr = statistics.stdev(cr_list)
    stdv_crs.append(stdv_cr)

    average_epv = statistics.mean(epv_list)
    average_epvs.append(round(average_epv*100,2))
    
    stdv_epv = statistics.stdev(epv_list)
    stdv_epvs.append(stdv_epv)
    
    ctr_min_bound = average_ctr - stdv_ctr
    ctrs_min_bound.append(round(ctr_min_bound*100,2))

    ctr_max_bound = average_ctr + stdv_ctr
    ctrs_max_bound.append(round(ctr_max_bound*100,2))

    cr_min_bound = average_cr - stdv_cr
    crs_min_bound.append(round(cr_min_bound*100,2))
    
    cr_max_bound = average_cr + stdv_cr
    crs_max_bound.append(round(cr_max_bound*100,2))
    
    epv_min_bound = average_epv - stdv_epv
    epvs_min_bound.append(round(epv_min_bound*100,2))
    
    epv_max_bound = average_epv + stdv_epv
    epvs_max_bound.append(round(epv_max_bound*100,2))


col1, col2, col3 = st.columns(3)

    
## Arranging CTR data

ctr_data = pd.DataFrame({'dates': date_list, 'ctr_avgs': average_ctrs, 'ctr_stdvs': stdv_ctrs, 'ctr_min_bound': ctrs_min_bound, 'ctr_max_bound': ctrs_max_bound})

ctr_data['day'] = np.arange(len(ctr_data))

ctr_data['prev_max_bound'] = ctr_data['ctr_max_bound'].shift(1)
ctr_data['diff_max_bound'] = abs(ctr_data['prev_max_bound'] - ctr_data['ctr_max_bound'])
ctr_data['diff_less'] = ctr_data['diff_max_bound'] < ctr_parameter
#ctr_data['prev_diff_less'] = ctr_data['diff_less'].shift(1)
#ctr_data['twice_diff']=ctr_data[['diff_less', 'prev_diff_less']].apply(lambda x: 1 if x[0] and x[1] else 0, axis=1)
#if len(ctr_data[['day']][ctr_data['twice_diff'] == 1]) == 0:
#    ctr_day_of_stability = 0
#else:
#    ctr_day_of_stability = int(ctr_data[['day']][ctr_data['twice_diff'] == 1].min())


with col1:
    st.header("CTR over Time")

    st.write(f"After one week we expect an interval between {ctr_data.iloc[6]['ctr_min_bound']}% and {ctr_data.iloc[6]['ctr_max_bound']}%")

    st.write(f"After two weeks we expect an interval between {ctr_data.iloc[13]['ctr_min_bound']}% and {ctr_data.iloc[13]['ctr_max_bound']}%")

    ## creating the graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ctr_data['day'], y=ctr_data['ctr_avgs'],
        fill=None,
        mode='lines',
        line_color='Black',
        name='Average'
        ))
    fig.add_trace(go.Scatter(
        x=ctr_data['day'],
        y=ctr_data['ctr_min_bound'],
        fillcolor='red',
        fill='tonexty', # fill area between trace0 and trace1
        mode='lines', line_color='indigo',
        name='Lower Bound'
        ))
        
    fig.add_trace(go.Scatter(
        x=ctr_data['day'],
        y=ctr_data['ctr_max_bound'],
        #fillcolor='rgb(128,177, 211)',
        fill='tonexty', # fill area between trace0 and trace1
        mode='lines', line_color='blue',
        name='Upper Bound'))

    fig.update_layout(
        autosize=False,
        width=800,
        height=490)
    fig.update_xaxes(title_text='Day')
    fig.update_yaxes(title_text='Running AA confidence interval')
    fig.update_yaxes(ticksuffix='%')
    #fig.add_vline(x=ctr_day_of_stability, line_width=3, line_dash="dash", line_color="white")
    st.plotly_chart(fig, use_container_width=True)

## Arranging CR data

with col2:
    st.header("CR over Time")


    cr_data = pd.DataFrame({'dates': date_list, 'cr_avgs': average_crs, 'cr_stdvs': stdv_crs, 'cr_min_bound': crs_min_bound, 'cr_max_bound': crs_max_bound})
    cr_data['day'] = np.arange(len(cr_data))

    cr_data['prev_max_bound'] = cr_data['cr_max_bound'].shift(1)
    cr_data['diff_max_bound'] = abs(cr_data['prev_max_bound'] - cr_data['cr_max_bound'])
    cr_data['diff_less'] = cr_data['diff_max_bound'] < cr_parameter
    #cr_data['prev_diff_less'] = cr_data['diff_less'].shift(1)
    #cr_data['twice_diff']=cr_data[['diff_less', 'prev_diff_less']].apply(lambda x: 1 if x[0] and x[1] else 0, axis=1)
    #if len(cr_data[['day']][cr_data['twice_diff'] == 1]) == 0:
    #    cr_day_of_stability = 0
    #else:
    #    cr_day_of_stability = int(cr_data[['day']][cr_data['twice_diff'] == 1].min())



    st.write(f"After one week we expect an interval between {cr_data.iloc[6]['cr_min_bound']}% and {cr_data.iloc[6]['cr_max_bound']}%")

    st.write(f"After two weeks we expect an interval between {cr_data.iloc[13]['cr_min_bound']}% and {cr_data.iloc[13]['cr_max_bound']}%")

    ## CR figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cr_data['day'], y=cr_data['cr_avgs'],
        fill=None,
        mode='lines',
        line_color='Black',
        name='Average'
        ))
    fig.add_trace(go.Scatter(
        x=cr_data['day'],
        y=cr_data['cr_min_bound'],
        fillcolor='red',
        fill='tonexty', # fill area between trace0 and trace1
        mode='lines', line_color='indigo',
        name='Lower Bound'
        ))
        
    fig.add_trace(go.Scatter(
        x=cr_data['day'],
        y=cr_data['cr_max_bound'],
        #fillcolor='rgb(128,177, 211)',
        fill='tonexty', # fill area between trace0 and trace1
        mode='lines', line_color='blue',
        name='Upper Bound'))

    fig.update_layout(
        autosize=False,
        width=800,
        height=490)
    fig.update_xaxes(title_text='Day')
    fig.update_yaxes(title_text='Running AA confidence interval')
    fig.update_yaxes(ticksuffix='%')
    #fig.add_vline(x=cr_day_of_stability, line_width=3, line_dash="dash", line_color="white")
    st.plotly_chart(fig, use_container_width=True)


## Arranging EPV

with col3:
    st.header("EPV over Time")

    epv_data = pd.DataFrame({'dates': date_list, 'epv_avgs': average_epvs, 'epv_stdvs': stdv_epvs, 'epv_min_bound': epvs_min_bound, 'epv_max_bound': epvs_max_bound})
    epv_data['day'] = np.arange(len(epv_data))

    epv_data['prev_max_bound'] = epv_data['epv_max_bound'].shift(1)
    epv_data['diff_max_bound'] = abs(epv_data['prev_max_bound'] - epv_data['epv_max_bound'])
    epv_data['diff_less'] = epv_data['diff_max_bound'] < epv_parameter
    #epv_data['prev_diff_less'] = epv_data['diff_less'].shift(1)
    #epv_data['twice_diff']=epv_data[['diff_less', 'prev_diff_less']].apply(lambda x: 1 if x[0] and x[1] else 0, axis=1)
    #if len(epv_data[['day']][epv_data['twice_diff'] == 1]) == 0:
   #     epv_day_of_stability = 0
   # else:
    #    epv_day_of_stability = int(epv_data[['day']][epv_data['twice_diff'] == 1].min())



    st.write(f"After one week we expect an interval between {epv_data.iloc[6]['epv_min_bound']}% and {epv_data.iloc[6]['epv_max_bound']}%")

    st.write(f"After two weeks we expect an interval between {epv_data.iloc[13]['epv_min_bound']}% and {epv_data.iloc[13]['epv_max_bound']}%")

    ## EPV graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epv_data['day'], y=epv_data['epv_avgs'],
        fill=None,
        mode='lines',
        line_color='Black',
        name='Average'
        ))
    fig.add_trace(go.Scatter(
        x=epv_data['day'],
        y=epv_data['epv_min_bound'],
        fillcolor='red',
        fill='tonexty', # fill area between trace0 and trace1
        mode='lines', line_color='indigo',
        name='Lower Bound'
        ))
        
    fig.add_trace(go.Scatter(
        x=epv_data['day'],
        y=epv_data['epv_max_bound'],
        #fillcolor='rgb(128,177, 211)',
        fill='tonexty', # fill area between trace0 and trace1
        mode='lines', line_color='blue',
        name='Upper Bound'))

    fig.update_layout(
        autosize=False,
        width=800,
        height=490)
    fig.update_xaxes(title_text='Day')
    fig.update_yaxes(title_text='Running AA confidence interval')
    fig.update_yaxes(ticksuffix='%')
    #fig.add_vline(x=epv_day_of_stability, line_width=3, line_dash="dash", line_color="white")
    st.plotly_chart(fig, use_container_width=True)
