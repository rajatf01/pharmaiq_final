"""
supplier_module/dashboard/supplier_tab.py
Fixed: robust path resolution so imports work whether called from
       dashboard/app.py or standalone.
"""
import os, sys, numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

_HERE     = os.path.dirname(os.path.abspath(__file__))
_SUPPLIER = os.path.dirname(_HERE)
_ROOT     = os.path.dirname(_SUPPLIER)
for _p in [_SUPPLIER, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from supplier_module.data.supplier_data import (
        generate_synthetic_supplier_products, generate_synthetic_products,
        SUPPLIER_ID, SUPPLIER_NAME, PRODUCT_ID, PRODUCT_NAME, CATEGORY,
        UNIT_COST, STATED_LEAD_TIME, DEFECT_RATE, COMPLIANCE_SCORE,
        FULFILLMENT_RATE, DELAY_PROBABILITY, AVG_DELAY_DAYS,
        RETURNS_RATE, BUY_VS_USAGE, MIN_SHELF_LIFE, TOTAL_ORDERS, LEAD_TIME_CV,
    )
    from supplier_module.models.scoring_engine import (
        score_all_suppliers, compute_tradeoff, DEFAULT_WEIGHTS,
        RISK_SCORE, RISK_LABEL, MCDM_SCORE, COST_RANK, MCDM_RANK,
        IS_RECOMMENDED, RECOMMENDATION_CONF, EFF_LEAD_TIME,
    )
except ImportError:
    from supplier_module.data.supplier_data import (
        generate_synthetic_supplier_products, generate_synthetic_products,
        SUPPLIER_ID, SUPPLIER_NAME, PRODUCT_ID, PRODUCT_NAME, CATEGORY,
        UNIT_COST, STATED_LEAD_TIME, DEFECT_RATE, COMPLIANCE_SCORE,
        FULFILLMENT_RATE, DELAY_PROBABILITY, AVG_DELAY_DAYS,
        RETURNS_RATE, BUY_VS_USAGE, MIN_SHELF_LIFE, TOTAL_ORDERS, LEAD_TIME_CV,
    )
    from supplier_module.models.scoring_engine import (
        score_all_suppliers, compute_tradeoff, DEFAULT_WEIGHTS,
        RISK_SCORE, RISK_LABEL, MCDM_SCORE, COST_RANK, MCDM_RANK,
        IS_RECOMMENDED, RECOMMENDATION_CONF, EFF_LEAD_TIME,
    )

C_GREEN="#00C853"; C_AMBER="#FFB300"; C_RED="#FF1744"
C_BLUE="#0091EA";  C_PURPLE="#AA00FF"
RISK_COLOURS={"Low":C_GREEN,"Medium":C_AMBER,"High":C_RED,"Critical":C_PURPLE}

SUPPLIER_CSS="""
<style>
.sup-header{background:linear-gradient(135deg,#0A0E1A,#131829,#0D1525);
  border:1px solid #1E2D50;border-radius:16px;padding:24px 28px;margin-bottom:20px}
.sup-header h2{color:#E8F0FE;font-size:22px;font-weight:700;margin:0 0 4px 0}
.sup-header p{color:#8FA3BF;font-size:13px;margin:0}
.sup-card{background:#131829;border-radius:14px;padding:18px 16px;
  border:1.5px solid #1E2D50;margin-bottom:8px;position:relative;overflow:hidden}
.sup-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,#0091EA,#00BFA5)}
.sup-card.recommended::before{background:linear-gradient(90deg,#00C853,#69F0AE)}
.sup-card.expensive::before{background:linear-gradient(90deg,#FF1744,#FF6D00)}
.sup-rank{font-size:11px;color:#8FA3BF;letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px}
.sup-name{font-size:16px;font-weight:700;color:#E8F0FE;margin-bottom:10px}
.sup-cost{font-size:32px;font-weight:800;margin-bottom:6px}
.sup-cost span{font-size:13px;font-weight:400;color:#8FA3BF}
.sup-badges{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.badge{padding:3px 8px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid}
.badge-green{color:#00C853;border-color:#00C853;background:rgba(0,200,83,.1)}
.badge-amber{color:#FFB300;border-color:#FFB300;background:rgba(255,179,0,.1)}
.badge-red{color:#FF1744;border-color:#FF1744;background:rgba(255,23,68,.1)}
.badge-blue{color:#0091EA;border-color:#0091EA;background:rgba(0,145,234,.1)}
.badge-rec{color:#00C853;border-color:#00C853;background:rgba(0,200,83,.12);font-size:10px;padding:2px 7px}
.verdict-banner{border-radius:10px;padding:14px 20px;margin:12px 0;display:flex;align-items:center;gap:14px}
.verdict-banner.rec{background:rgba(0,200,83,.12);border:1.5px solid #00C853}
.verdict-banner.no{background:rgba(0,145,234,.10);border:1.5px solid #0091EA}
.verdict-icon{font-size:28px}
.verdict-title{font-size:16px;font-weight:700;color:#E8F0FE}
.verdict-sub{font-size:12px;color:#8FA3BF;margin-top:2px}
.factor-item{display:flex;align-items:flex-start;gap:10px;padding:8px 0;
  border-bottom:1px solid #1A2035;font-size:13px;color:#C5D3E8}
.factor-item:last-child{border-bottom:none}
.factor-dot{font-size:16px;flex-shrink:0;margin-top:1px}
.tradeoff-box{background:#0D1A2D;border:1px solid #1E3A5F;border-radius:12px;
  padding:16px 18px;font-family:monospace;font-size:12px;color:#90CAF9;
  line-height:1.9;white-space:pre-wrap}
.sup-divider{border:none;border-top:1px solid #1E2D50;margin:20px 0}
</style>
"""

def _badge(text, level):
    cls={"low":"badge-green","medium":"badge-amber","high":"badge-red",
         "blue":"badge-blue","rec":"badge-rec"}.get(level.lower(),"badge-blue")
    return f'<span class="badge {cls}">{text}</span>'

def _risk_c(s):
    return C_GREEN if s<0.35 else (C_AMBER if s<0.60 else C_RED)

def _cost_c(rank,n):
    return C_GREEN if rank==1 else (C_RED if rank==n else C_BLUE)

def safe(row,col,default=0):
    try:
        v=row.get(col,default)
        return default if (v is None or (isinstance(v,float) and np.isnan(v))) else v
    except: return default

@st.cache_data(show_spinner=False)
def _load():
    return score_all_suppliers(generate_synthetic_supplier_products(), DEFAULT_WEIGHTS.copy())

def render_supplier_tab():
    st.markdown(SUPPLIER_CSS, unsafe_allow_html=True)
    st.markdown("""<div class="sup-header">
      <h2>🏭 Supplier Intelligence &amp; Comparison</h2>
      <p>AI-powered multi-criteria ranking · Cost-first shortlist · Trade-off analysis · Explainable recommendations</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3 = st.columns([2.5,1.5,1])
    with c1:
        prod_df=generate_synthetic_products()
        opts=dict(zip(prod_df[PRODUCT_NAME],prod_df[PRODUCT_ID]))
        sel_name=st.selectbox("🔬 Select Product / Medicine",list(opts.keys()))
        sel_pid=opts[sel_name]
    with c2:
        sort_by=st.radio("📊 Rank by",["💰 Cost","⭐ AI Score","⏱ Lead Time","🎯 Reliability"],horizontal=False)
    with c3:
        order_qty=st.number_input("📦 Order Qty",min_value=100,max_value=10000,value=1000,step=100)

    with st.spinner("⚙️ Running AI scoring…"):
        all_df=_load()
    df=all_df[all_df[PRODUCT_ID]==sel_pid].copy()
    if df.empty:
        st.error(f"No suppliers for {sel_name}."); return

    sm={"💰 Cost":(UNIT_COST,True),"⭐ AI Score":(MCDM_SCORE,False),
        "⏱ Lead Time":(STATED_LEAD_TIME,True),"🎯 Reliability":(FULFILLMENT_RATE,False)}
    sc,sa=sm[sort_by]
    df=df.sort_values(sc,ascending=sa).reset_index(drop=True)
    n=len(df)
    rec_n=df[SUPPLIER_NAME][df[IS_RECOMMENDED]==True].values[0] if any(df[IS_RECOMMENDED]) else "—"
    min_c,max_c=df[UNIT_COST].min(),df[UNIT_COST].max()

    st.markdown(f"""<div style="background:#0D1525;border:1px solid #1E3058;border-radius:10px;
      padding:10px 18px;margin:10px 0 18px 0;display:flex;gap:32px;align-items:center;flex-wrap:wrap">
      <span style="color:#8FA3BF;font-size:12px"><strong style="color:#E8F0FE">{n}</strong> suppliers shortlisted</span>
      <span style="color:#8FA3BF;font-size:12px">Price: <strong style="color:#E8F0FE">${min_c:.2f}</strong> – <strong style="color:#E8F0FE">${max_c:.2f}</strong>/unit</span>
      <span style="color:#8FA3BF;font-size:12px">AI Pick: <strong style="color:#00C853">{rec_n}</strong></span>
    </div>""", unsafe_allow_html=True)

    cols=st.columns(min(n,3))
    rl_map={1:"🥇 CHEAPEST",2:"🥈 2ND",3:"🥉 3RD"}
    for i,(_,row) in enumerate(df.iterrows()):
        cr=int(row.get(COST_RANK,i+1)); is_r=bool(row.get(IS_RECOMMENDED,False))
        rl=str(row.get(RISK_LABEL,"Medium")); mcdm=float(row.get(MCDM_SCORE,70))
        delay=float(row.get(DELAY_PROBABILITY,0.3)); fill=float(row.get(FULFILLMENT_RATE,0.95))
        cost_c=_cost_c(cr,n); card_cls="recommended" if is_r else ("expensive" if cr==n else "")
        dl=lambda v,lo,hi: "green" if v<lo else ("amber" if v<hi else "red")
        with cols[i%len(cols)]:
            st.markdown(f"""<div class="sup-card {card_cls}">
              <div class="sup-rank">{rl_map.get(cr,f'#{cr}')}</div>
              <div class="sup-name">{row[SUPPLIER_NAME]}</div>
              <div class="sup-cost" style="color:{cost_c}">${float(row[UNIT_COST]):.2f}<span>/unit</span></div>
              <div class="sup-badges">
                {_badge(f"⏱ {int(row[STATED_LEAD_TIME])}d","blue")}
                {_badge(f"⭐ {mcdm:.0f}/100","blue")}
                {_badge(f"⚡ {delay:.0%} delay",dl(delay,.25,.50))}
                {_badge(f"📦 {fill:.0%} fill","green" if fill>.96 else "amber")}
                {_badge(f"🛡 {rl} risk","green" if rl=="Low" else ("amber" if rl=="Medium" else "red"))}
                {_badge("✅ AI PICK","rec") if is_r else ""}
              </div></div>""", unsafe_allow_html=True)

    st.markdown('<hr class="sup-divider"/>', unsafe_allow_html=True)

    sel_sup=st.selectbox("🔍 Drill into a supplier for full AI analysis",df[SUPPLIER_NAME].tolist(),index=0)
    sel=df[df[SUPPLIER_NAME]==sel_sup].iloc[0]
    is_r=bool(sel.get(IS_RECOMMENDED,False)); conf=float(sel.get(RECOMMENDATION_CONF,0.70))
    rl=str(sel.get(RISK_LABEL,"Medium")); mr=int(sel.get(MCDM_RANK,1))

    if is_r:
        st.markdown(f"""<div class="verdict-banner rec"><div class="verdict-icon">✅</div>
          <div><div class="verdict-title">Primary Recommendation — {sel_sup}</div>
          <div class="verdict-sub">Confidence: {conf:.0%} · MCDM Score: {float(sel.get(MCDM_SCORE,70)):.1f}/100 · Risk: {rl}</div>
          </div></div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="verdict-banner no"><div class="verdict-icon">#{mr}</div>
          <div><div class="verdict-title">Ranked #{mr} Overall — {sel_sup}</div>
          <div class="verdict-sub">MCDM Score: {float(sel.get(MCDM_SCORE,70)):.1f}/100 · Risk: {rl}</div>
          </div></div>""", unsafe_allow_html=True)

    dt1,dt2,dt3,dt4=st.tabs(["💰 Cost & Pricing","⏱ Lead Time","🧬 Quality","🤖 AI Intelligence"])

    with dt1:
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Unit Cost",f"${float(sel[UNIT_COST]):.2f}",delta="Cheapest ✅" if int(sel.get(COST_RANK,1))==1 else None)
        m2.metric("Min Order",f"{int(sel.get('min_order_qty',500)):,} units")
        m3.metric("Payment","Net 30"); m4.metric("Currency","USD")
        fp=px.bar(df.sort_values(UNIT_COST),x=SUPPLIER_NAME,y=UNIT_COST,
                  color=SUPPLIER_NAME,color_discrete_sequence=[C_GREEN,C_BLUE,C_AMBER,C_PURPLE],
                  title="Unit Cost Comparison",text=df.sort_values(UNIT_COST)[UNIT_COST].apply(lambda x:f"${x:.2f}"))
        fp.update_traces(textposition="outside")
        fp.update_layout(height=260,showlegend=False,paper_bgcolor="rgba(0,0,0,0)",
          plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#C5D3E8"),margin=dict(t=40,b=10),
          xaxis=dict(gridcolor="#1E2D50"),yaxis=dict(gridcolor="#1E2D50"))
        st.plotly_chart(fp,use_container_width=True)

    with dt2:
        eff_lt=float(sel.get(EFF_LEAD_TIME,sel[STATED_LEAD_TIME]))
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Stated Lead",f"{int(sel[STATED_LEAD_TIME])} days")
        m2.metric("Effective Lead",f"{eff_lt:.1f} days",help="95th-percentile — use for safety stock")
        m3.metric("Delay Probability",f"{float(sel[DELAY_PROBABILITY]):.0%}",
                  delta=f"avg {float(sel.get(AVG_DELAY_DAYS,0)):.1f}d late",delta_color="inverse")
        m4.metric("Lead Time CV",f"{float(sel.get(LEAD_TIME_CV,0.2)):.2f}")
        lt=df[[SUPPLIER_NAME,STATED_LEAD_TIME]].copy()
        if EFF_LEAD_TIME in df.columns: lt["Effective"]=df[EFF_LEAD_TIME]
        fl=go.Figure()
        fl.add_bar(name="Stated",x=lt[SUPPLIER_NAME],y=lt[STATED_LEAD_TIME],marker_color=C_BLUE,opacity=0.7)
        if "Effective" in lt.columns:
            fl.add_bar(name="Effective (95th pct)",x=lt[SUPPLIER_NAME],y=lt["Effective"],marker_color=C_AMBER,opacity=0.85)
        fl.update_layout(barmode="group",height=260,title="Stated vs Effective Lead Times",
          paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#C5D3E8"),
          margin=dict(t=40,b=10),legend=dict(orientation="h",y=1.15,font=dict(color="#C5D3E8"),bgcolor="rgba(0,0,0,0)"),
          xaxis=dict(gridcolor="#1E2D50"),yaxis=dict(gridcolor="#1E2D50",title="Days"))
        st.plotly_chart(fl,use_container_width=True)

    with dt3:
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Defect Rate",f"{float(sel[DEFECT_RATE]):.1%}")
        m2.metric("Compliance",f"{int(sel[COMPLIANCE_SCORE])}/100")
        m3.metric("Min Shelf Life",f"{int(sel[MIN_SHELF_LIFE])} days")
        m4.metric("Cold Chain","✅ Certified")
        dims=["Cost","Lead Time","Quality","Reliability","Shelf Life"]
        def rv(row):
            mc,xc=df[UNIT_COST].min(),df[UNIT_COST].max()
            ml,xl=df[STATED_LEAD_TIME].min(),df[STATED_LEAD_TIME].max()
            cs=1-(float(row[UNIT_COST])-mc)/(xc-mc+.01)
            ls=1-(float(row[STATED_LEAD_TIME])-ml)/(xl-ml+.01)
            qs=(float(row[COMPLIANCE_SCORE])/100)*(1-float(row[DEFECT_RATE])/0.05)
            rs=float(row[FULFILLMENT_RATE])*(1-float(row[DELAY_PROBABILITY]))
            ss=min(1.0,float(row[MIN_SHELF_LIFE])/365)
            return [cs*100,ls*100,qs*100,rs*100,ss*100]
        fr=go.Figure()
        for _,row in df.iterrows():
            v=rv(row); issel=row[SUPPLIER_NAME]==sel_sup
            fr.add_trace(go.Scatterpolar(r=v+[v[0]],theta=dims+[dims[0]],fill="toself",
              name=row[SUPPLIER_NAME],line=dict(width=3 if issel else 1,color=C_GREEN if issel else "#2D3045"),
              fillcolor="rgba(0,200,83,0.15)" if issel else "rgba(45,48,69,0.3)",opacity=1 if issel else 0.5))
        fr.update_layout(polar=dict(bgcolor="#0D1525",
          radialaxis=dict(visible=True,range=[0,100],gridcolor="#1E2D50",tickfont=dict(color="#8FA3BF")),
          angularaxis=dict(gridcolor="#1E2D50",tickfont=dict(color="#C5D3E8"))),
          showlegend=True,height=320,paper_bgcolor="rgba(0,0,0,0)",
          margin=dict(t=20,b=20,l=40,r=40),legend=dict(font=dict(color="#C5D3E8"),bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fr,use_container_width=True)

    with dt4:
        rs=float(sel.get(RISK_SCORE,0.4))
        m1,m2,m3,m4=st.columns(4)
        m1.metric("AI Risk Score",f"{rs:.2f}/1.00",delta=rl,
                  delta_color="inverse" if rl in ["High","Critical"] else "off")
        m2.metric("Fulfillment",f"{float(sel[FULFILLMENT_RATE]):.1%}")
        m3.metric("Buy vs Usage",f"{float(sel.get(BUY_VS_USAGE,0.94)):.1%}")
        m4.metric("Orders",int(sel.get(TOTAL_ORDERS,0)))
        fg=go.Figure(go.Indicator(mode="gauge+number+delta",value=rs*100,
          delta={"reference":40,"decreasing":{"color":C_GREEN},"increasing":{"color":C_RED}},
          title={"text":"AI Risk Score (0=Safe · 100=Dangerous)","font":{"color":"#C5D3E8","size":13}},
          gauge={"axis":{"range":[0,100],"tickfont":{"color":"#8FA3BF"}},
                 "bar":{"color":_risk_c(rs)},"bgcolor":"#0D1525","bordercolor":"#1E2D50",
                 "steps":[{"range":[0,35],"color":"rgba(0,200,83,.15)"},
                           {"range":[35,60],"color":"rgba(255,179,0,.12)"},
                           {"range":[60,100],"color":"rgba(255,23,68,.15)"}],
                 "threshold":{"line":{"color":"#fff","width":2},"thickness":.75,"value":rs*100}},
          number={"suffix":"/100","font":{"color":"#E8F0FE","size":28}}))
        fg.update_layout(height=240,paper_bgcolor="rgba(0,0,0,0)",
          font=dict(color="#C5D3E8"),margin=dict(t=30,b=10,l=20,r=20))
        st.plotly_chart(fg,use_container_width=True)
        positives=list(sel.get("positives",[])); risks_=list(sel.get("risks",[]))
        if positives or risks_:
            xa,xb=st.columns(2)
            with xa:
                st.markdown("**✅ Positive factors**")
                for pt in positives:
                    st.markdown(f'<div class="factor-item"><span class="factor-dot" style="color:{C_GREEN}">●</span><span>{pt}</span></div>',unsafe_allow_html=True)
            with xb:
                st.markdown("**⚠️ Risk factors**")
                for rk in risks_:
                    st.markdown(f'<div class="factor-item"><span class="factor-dot" style="color:{C_AMBER}">●</span><span>{rk}</span></div>',unsafe_allow_html=True)
        act=sel.get("action","")
        if act:
            st.markdown(f"""<div style="background:#0A1929;border:1px solid #0091EA;border-radius:10px;
              padding:12px 16px;margin-top:12px"><span style="color:#90CAF9;font-size:12px;
              font-weight:600;letter-spacing:.06em;text-transform:uppercase">Recommended Action</span>
              <div style="color:#E8F0FE;font-size:14px;margin-top:4px">{act}</div></div>""",unsafe_allow_html=True)

    st.markdown('<hr class="sup-divider"/>',unsafe_allow_html=True)
    st.markdown("**🎯 Cost vs AI Score — Supplier Positioning Map**")
    st.caption("Lower cost + higher AI Score = ideal. Bubble size = fulfillment rate.")
    fs=px.scatter(df,x=UNIT_COST,y=MCDM_SCORE,color=RISK_LABEL,size=FULFILLMENT_RATE,size_max=35,
      hover_name=SUPPLIER_NAME,hover_data={UNIT_COST:":.2f",MCDM_SCORE:":.1f",RISK_SCORE:":.3f",DELAY_PROBABILITY:":.0%"},
      color_discrete_map={"Low":C_GREEN,"Medium":C_AMBER,"High":C_RED,"Critical":C_PURPLE},
      labels={UNIT_COST:"Unit Cost (USD/unit)",MCDM_SCORE:"AI Score (0–100)",RISK_LABEL:"Risk"})
    for _,r in df.iterrows():
        if bool(r.get(IS_RECOMMENDED,False)):
            fs.add_annotation(x=float(r[UNIT_COST]),y=float(r[MCDM_SCORE]),text="✅ Best Pick",
              showarrow=True,arrowhead=2,font=dict(color=C_GREEN,size=11),arrowcolor=C_GREEN)
    fs.update_layout(height=340,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0D1525",
      font=dict(color="#C5D3E8"),margin=dict(t=20,b=30),
      xaxis=dict(gridcolor="#1E2D50",zerolinecolor="#1E2D50"),
      yaxis=dict(gridcolor="#1E2D50",zerolinecolor="#1E2D50"),
      legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#C5D3E8")))
    st.plotly_chart(fs,use_container_width=True)

    st.markdown('<hr class="sup-divider"/>',unsafe_allow_html=True)
    st.markdown("**💰 Financial Trade-Off Calculator**")
    if n>=2:
        tc1,tc2=st.columns(2)
        with tc1: a_name=st.selectbox("Supplier A",df[SUPPLIER_NAME].tolist(),index=0,key="tfa")
        with tc2: b_name=st.selectbox("vs Supplier B",df[SUPPLIER_NAME].tolist(),index=min(1,n-1),key="tfb")
        if a_name!=b_name:
            sa2=df[df[SUPPLIER_NAME]==a_name].iloc[0]; sb2=df[df[SUPPLIER_NAME]==b_name].iloc[0]
            tf=compute_tradeoff(sa2,sb2,order_qty=int(order_qty))
            k1,k2,k3,k4=st.columns(4)
            k1.metric("Cost Saving (A vs B)",f"${tf['cost_saving']:+.2f}")
            k2.metric("Delay Cost Premium",f"-${tf['delay_cost']:.2f}")
            k3.metric("Quality Cost Diff",f"-${tf['quality_cost_diff']:.2f}")
            k4.metric("Net Benefit (A over B)",f"${tf['net_benefit']:+.2f}",
                      delta="Choose A ✅" if tf["net_benefit"]>=0 else "Choose B ✅",
                      delta_color="normal" if tf["net_benefit"]>=0 else "inverse")
            nb=tf["net_benefit"]; win=tf["recommended"]; col=C_GREEN if nb>=0 else C_AMBER
            st.markdown(f"""<div style="background:rgba(0,145,234,.08);border:1px solid #0091EA;
              border-radius:10px;padding:14px 18px;margin-top:8px">
              <span style="color:#90CAF9;font-weight:700;font-size:14px">
                🏆 Choose <strong style="color:{col}">{win}</strong> —
                net saving of <strong style="color:{col}">${abs(nb):.2f}</strong>
                per {int(order_qty):,}-unit order.</span></div>""",unsafe_allow_html=True)
            txt=tf.get("text","")
            if txt:
                with st.expander("📋 Detailed breakdown"):
                    st.markdown(f'<div class="tradeoff-box">{txt}</div>',unsafe_allow_html=True)

    st.markdown('<hr class="sup-divider"/>',unsafe_allow_html=True)
    with st.expander("📊 Full Side-by-Side Comparison Table"):
        cc=[SUPPLIER_NAME,UNIT_COST,STATED_LEAD_TIME]+([EFF_LEAD_TIME] if EFF_LEAD_TIME in df.columns else [])+\
           [DEFECT_RATE,COMPLIANCE_SCORE,DELAY_PROBABILITY,FULFILLMENT_RATE,RISK_SCORE,MCDM_SCORE]
        av=[c for c in cc if c in df.columns]; cmp=df[av].copy()
        cmp=cmp.rename(columns={SUPPLIER_NAME:"Supplier",UNIT_COST:"Cost ($)",STATED_LEAD_TIME:"Lead (d)",
          EFF_LEAD_TIME:"Eff Lead (d)",DEFECT_RATE:"Defect %",COMPLIANCE_SCORE:"Compliance",
          DELAY_PROBABILITY:"Delay %",FULFILLMENT_RATE:"Fulfill %",RISK_SCORE:"Risk Score",MCDM_SCORE:"AI Score"})
        for c in ["Defect %","Delay %","Fulfill %"]:
            if c in cmp.columns: cmp[c]=cmp[c].apply(lambda x:f"{float(x):.1%}")
        if "Cost ($)" in cmp.columns: cmp["Cost ($)"]=cmp["Cost ($)"].apply(lambda x:f"${float(x):.2f}")
        st.dataframe(cmp,use_container_width=True,hide_index=True)
        st.download_button("⬇️ Export CSV",cmp.to_csv(index=False).encode(),
                           f"supplier_comparison_{sel_pid}.csv","text/csv")

if __name__=="__main__":
    st.set_page_config(page_title="Supplier Intelligence",page_icon="🏭",layout="wide")
    render_supplier_tab()
