import {YoutubeVideo} from '@google/glue/lib/ytvideo';
import {BehaviorSubject, first} from 'rxjs';
import events, {Events} from '../events/events';
import {Video} from './video.d';

const enum Selectors {
  video = '.glue-video',
  glueVideoContainer = '.glue-video__container',
}

module VideoModule {
  const videos: {[key: string]: Video} = {};
  const ytApiLoaded: BehaviorSubject<boolean> = new BehaviorSubject(false);

  let inited = false;

  function init(): void {
    ytInit().then(() => initPlayers());
  }

  function ytInit(): Promise<void> {
    return new Promise(resolve => {
      if (!inited) loadScript();
      inited = true;
      ytApiLoaded
        .asObservable()
        .pipe(first(val => val))
        .subscribe(() => resolve());
    });
  }

  function loadScript(): void {
    const script = document.createElement('script');
    script.onload = () => ytApiLoaded.next(true);
    script.src = 'https://www.youtube.com/iframe_api';
    document.head.appendChild(script);
  }

  function initPlayers(): void {
    const videoEls: NodeListOf<HTMLElement> = document.querySelectorAll(Selectors.video);
    for (const el of videoEls) addVideo(el);
  }

  function addVideo(el: HTMLElement): void {
    if (!el.querySelector(Selectors.glueVideoContainer)) return;
    if (!el.id) el.id = `video-${Object.entries(videos).length}`;
    const id = el.id;
    const newVideo = {
      container: el,
      player: new YoutubeVideo(el),
    };
    bindEvents(newVideo);
    videos[id] = newVideo;
  }

  function bindEvents(video: Video): void {
    // pause player if its parent carousel changes
    events.addEventListener(Events.carouselSlideChanged, ((e: CustomEvent) => {
      if (video.container.closest(`#${e.detail.carouselId}`)) {
        const player = video.player.getPlayer();
        if (player) player.pauseVideo();
      }
    }) as EventListener);
  }

  init();
}
