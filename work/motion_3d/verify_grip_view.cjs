const { chromium } = require('C:/Users/junwoo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({headless:true, executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
  try {
    const page = await browser.newPage({viewport:{width:820,height:820}});
    const errors = [];
    page.on('pageerror',e=>errors.push(String(e)));
    await page.goto('file:///C:/Users/junwoo/Documents/Codex/2026-10-10/new-chat-2/work/motion_3d/preview.html');
    const frame = page.frameLocator('iframe');
    await frame.locator('#swing-motion-3d[data-ready="true"]').waitFor();
    const cases={};
    for(const name of ['rod-ideal','rod-with-resistance','rope-ideal','rope-with-resistance']){
      await frame.locator('[data-control="case"]').selectOption(name);
      const stats=await frame.locator('#swing-motion-3d').evaluate(root=>{
        const slider=root.querySelector('[data-control="time"]'),end=Number(slider.max),results=[];
        for(let i=0;i<=40;i++){
          slider.value=end*i/40;slider.dispatchEvent(new Event('input',{bubbles:true}));
          const theta=Number(root.dataset.angle)*Math.PI/180;
          for(const [index,arm] of root.armPoses.entries()){
            const [x,y,z]=arm.hand,c=Math.cos(theta),s=Math.sin(theta),localX=c*x+s*(z-2.5);
            const localZ=-s*x+c*(z-2.5),side=index===0?-1:1;
            const d=arm.hand.map((v,j)=>v-arm.shoulder[j]);
            const distance=Math.hypot(...d),direction=d.map(v=>v/distance);
            const along=(.32*.32-.34*.34+distance*distance)/(2*distance);
            const h=Math.sqrt(.32*.32-along*along),lowest=arm.shoulder[2]+along*direction[2]-h*Math.sqrt(Math.max(0,1-direction[2]**2));
            results.push({lengthError:Number(root.dataset.armLengthError),gripXError:Math.abs(localX),gripYError:Math.abs(y-side*.24),lowestError:Math.abs(arm.elbow[2]-lowest),angle:arm.angle,localZ});
          }
        }
        return {samples:results.length,maxLengthError:Math.max(...results.map(x=>x.lengthError)),maxGripXError:Math.max(...results.map(x=>x.gripXError)),maxGripYError:Math.max(...results.map(x=>x.gripYError)),maxLowestError:Math.max(...results.map(x=>x.lowestError)),angleRange:[Math.min(...results.map(x=>x.angle)),Math.max(...results.map(x=>x.angle))],end:Number(root.dataset.time),theta:Number(root.dataset.angle)};
      });
      if(Math.max(stats.maxLengthError,stats.maxGripXError,stats.maxGripYError,stats.maxLowestError)>1e-5)throw new Error('Arm constraint failure: '+JSON.stringify(stats));
      cases[name]=stats;
    }
    await frame.locator('[data-control="case"]').selectOption('rod-ideal');
    await frame.locator('[data-control="play"]').click();
    await frame.locator('#swing-motion-3d').evaluate(root=>new Promise(resolve=>{
      const observer=new MutationObserver(()=>{if(Number(root.dataset.time)>0.7){observer.disconnect();resolve();}});observer.observe(root,{attributes:true,attributeFilter:['data-time']});
    }));
    await frame.locator('[data-control="play"]').click();
    await frame.locator('[data-control="time"]').evaluate(el=>{el.value='12.5';el.dispatchEvent(new Event('input',{bubbles:true}));});
    await page.screenshot({path:path.join(__dirname,'grip-browser-desktop.png')});
    await page.setViewportSize({width:360,height:820});
    await page.emulateMedia({colorScheme:'dark'});
    await page.screenshot({path:path.join(__dirname,'grip-browser-mobile.png')});
    const overflow=await frame.locator('#swing-motion-3d').evaluate(root=>({width:root.clientWidth,scroll:root.scrollWidth,canvas:root.querySelector('canvas').getBoundingClientRect().width}));
    if(overflow.scroll>overflow.width+2)throw new Error('Narrow layout overflow');
    if(errors.length)throw new Error(errors.join('\n'));
    fs.writeFileSync(path.join(__dirname,'grip-browser-verification.json'),JSON.stringify({cases,overflow,errors},null,2));
    console.log(JSON.stringify({cases,overflow,errors},null,2));
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
