document.querySelectorAll('[data-menu]').forEach(button=>{
  button.addEventListener('click',()=>{
    const nav=document.querySelector('.nav');
    const open=nav.classList.toggle('open');
    button.setAttribute('aria-expanded',String(open));
  });
});

document.querySelectorAll('[data-copy]').forEach(button=>{
  button.addEventListener('click',async()=>{
    const block=document.getElementById(button.dataset.copy);
    if(!block)return;
    try{
      await navigator.clipboard.writeText(block.textContent);
      const previous=button.textContent;
      button.textContent='Copied';
      setTimeout(()=>button.textContent=previous,1600);
    }catch{button.textContent='Select code to copy';}
  });
});

const filters=[...document.querySelectorAll('[data-filter]')];
const rows=[...document.querySelectorAll('.archive-row[data-kind]')];
const count=document.getElementById('filter-count');
filters.forEach(button=>button.addEventListener('click',()=>{
  const selected=button.dataset.filter;
  filters.forEach(item=>item.setAttribute('aria-pressed',String(item===button)));
  let visible=0;
  rows.forEach(row=>{
    row.hidden=selected!=='all' && row.dataset.kind!==selected;
    if(!row.hidden)visible++;
  });
  if(count)count.textContent=`${String(visible).padStart(2,'0')} ${visible===1?'entry':'entries'}`;
}));
