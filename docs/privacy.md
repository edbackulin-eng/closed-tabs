---
title: Privacy Policy
---

# Privacy Policy — Closed Tabs

**Last updated: 14 October 2026**

## What this application is

The Closed Tabs uploader is a private, automated tool that its owner runs to
publish videos to a single YouTube channel they own. It is not a product, not a
service, and not offered to anyone else. There is no sign-up, no login screen
and no user interface. The source code is public so that what it does can be
checked; the running instance belongs to one person.

## Who uses it

One person: the owner of the channel it publishes to. There are no other users
and no mechanism by which anyone else could become one. Nothing in this policy
is a promise to third parties about their data, because the application never
receives any.

## What it collects

**Nothing from anybody.** The application has no users to collect data from. It
does not read viewer data, comments, subscriber information, analytics or any
other YouTube data about other people. It writes to one channel and reads back
only that channel's own name, to confirm it is authenticated against the right
account, and the publication status of the videos it uploaded itself.

## Credentials

Authorising the application produces an OAuth refresh token for the owner's own
Google account. That token is stored as an encrypted GitHub Actions secret in
the repository that runs the pipeline. It is never written to logs, never
committed to the repository and never sent anywhere other than Google's own API
endpoints.

The owner can revoke it at any time at
[myaccount.google.com/permissions](https://myaccount.google.com/permissions),
which immediately stops the application from being able to upload.

## What it sends to Google

Only what is needed to publish a video the owner made: the video file, its
thumbnail, its subtitle track, and its metadata — title, description, tags,
category, language, the "not made for kids" flag and the synthetic-media
disclosure.

## Sharing

No YouTube data is shared with third parties. None is sold. None is used to
train machine learning models of any kind, including the AI tools used to help
produce the videos themselves.

## Scopes and why each is needed

- `youtube.upload` — upload the video file
- `youtube.force-ssl` — set the thumbnail, subtitles and metadata
- `youtube.readonly` — read the channel's own name at setup, to confirm the
  application is pointed at the right channel

## Google API Services User Data Policy

This application's use of information received from Google APIs follows the
[Google API Services User Data Policy](https://developers.google.com/terms/api-services-user-data-policy),
including the Limited Use requirements.

## Changes

If this policy changes, the updated version appears on this page with a new
date.

## Contact

edwardcatstory@gmail.com
